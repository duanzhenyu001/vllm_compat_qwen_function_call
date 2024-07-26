import copy
import json
from typing import Dict, AsyncGenerator

from vllm.entrypoints.openai.protocol import (
    ChatCompletionResponse,
    DeltaMessage
)
from vllm.entrypoints.qwen.utils.schema import *
from vllm.entrypoints.qwen.utils.utils import format_as_multimodal_message, qwen_to_openai_response_message
from vllm.entrypoints.qwen.utils.tokenization_qwen import tokenizer
from vllm.logger import init_logger

logger = init_logger(__name__)


async def post_process_message_nostream(
        response: ChatCompletionResponse,
        fncall_mode: bool,
        generate_cfg: dict
):
    messages = [Message(role=ASSISTANT, content=response.choices[0].message.content)]
    qwen_message = await _postprocess_messages(messages, fncall_mode, generate_cfg)
    message = qwen_to_openai_response_message(qwen_message)
    response.choices[0].message = message
    return response


class SFnStatus:
    exact = 1
    not_sure = 0
    exact_not = -1


def check_function_str(s: str):
    if s.endswith(FN_NAME) or FN_NAME in s:
        return SFnStatus.exact
    for i in range(min(len(s), len(FN_NAME)), 0, -1):
        if FN_NAME.startswith(s[-i:]):
            return SFnStatus.not_sure
    return SFnStatus.exact_not


async def post_process_message_stream(
        response: AsyncGenerator[str, None],
        fncall_mode: bool,
        generate_cfg: dict
) -> AsyncGenerator[str, None]:
    full_response = ""
    last_chunk = {}
    fn_exist = False
    async for raw_chunk in response:
        try:
            if STREAM_END == raw_chunk:
                logger.info(f"stream end: {raw_chunk}")
                if full_response:
                    pre_msg = [Message(ASSISTANT, full_response)]
                    logger.info(f"stream full_response: {full_response}")
                    post_msg = await _postprocess_messages(pre_msg, fncall_mode=fncall_mode, generate_cfg=generate_cfg)
                    logger.info(f"stream post process message: {post_msg}")
                    openai_msg = qwen_to_openai_response_message(post_msg)
                    delta_message = DeltaMessage(tool_calls=openai_msg.tool_calls)
                    last_chunk["choices"][0]["delta"] = delta_message.model_dump(exclude_unset=True)
                    yield OPENAI_DATA_START + json.dumps(last_chunk, ensure_ascii=False)
                    yield raw_chunk
                else:
                    yield raw_chunk
                return

            chunk = json.loads(raw_chunk[len(OPENAI_DATA_START):])
            last_chunk = chunk
            if chunk["choices"][0]["delta"].get("content", ""):
                full_response += chunk["choices"][0]["delta"]["content"]

                if fn_exist:
                    chunk["choices"][0]["delta"]["content"] = ""
                    # 空响应 防止单次响应超时
                    yield OPENAI_DATA_START + json.dumps(chunk, ensure_ascii=False)

                fn_status = check_function_str(full_response)
                if fn_status == SFnStatus.not_sure:
                    continue
                elif fn_status == SFnStatus.exact:
                    fn_exist = True
                    if not full_response.startswith(FN_NAME):
                        splits = full_response.split(FN_NAME)
                        full_response = FN_NAME + splits[1]
                        chunk["choices"][0]["delta"]["content"] = splits[0]
                        # 将出现在function 之前的内容返回
                        yield OPENAI_DATA_START + json.dumps(chunk, ensure_ascii=False)
                else:
                    full_response = ""
                    yield raw_chunk
            else:
                yield raw_chunk
        except Exception as e:
            logger.info(f"raw_chunk: {raw_chunk} error: {e}, full_response: {full_response}")


async def _postprocess_messages(
        messages: List[Message],
        fncall_mode: bool,
        generate_cfg: dict,
) -> List[Message]:
    messages = [format_as_multimodal_message(msg, add_upload_info=False) for msg in messages]
    if not generate_cfg.get('skip_stopword_postproc', False):
        stop = generate_cfg.get('stop', [])
        stop += [x for x in FN_STOP_WORDS if x not in stop]
        messages = _postprocess_stop_words(messages, stop=stop)
    if fncall_mode:
        fn_choice = generate_cfg.get('function_choice', 'auto')
        if fn_choice not in ('auto', 'none'):
            messages = copy.deepcopy(messages)
            output = messages[0].content[0].text
            if output.lstrip().startswith(FN_ARGS):
                # Prepend this fn_choice prefix only if the model correctly completes it
                output = f'{FN_NAME}: {fn_choice}\n' + output
            messages[0].content[0].text = output
        messages = _postprocess_fncall_messages(messages)
    return messages


def _postprocess_stop_words(messages: List[Message], stop: List[str]) -> List[Message]:
    messages = copy.deepcopy(messages)

    # Make sure it stops before stop words.
    trunc_messages = []
    for msg in messages:
        truncated = False
        trunc_content = []
        for i, item in enumerate(msg.content):
            item_type, item_text = item.get_type_and_value()
            if item_type == 'text':
                truncated, item.text = _truncate_at_stop_word(text=item_text, stop=stop)
            trunc_content.append(item)
            if truncated:
                break
        msg.content = trunc_content
        trunc_messages.append(msg)
        if truncated:
            break
    messages = trunc_messages

    # It may ends with partial stopword 'Observation' when the full stopword is 'Observation:'.
    # The following post-processing step removes partial stop words.
    partial_stop = []
    for s in stop:
        s = tokenizer.tokenize(s)[:-1]
        if s:
            s = tokenizer.convert_tokens_to_string(s)
            partial_stop.append(s)
    partial_stop = sorted(set(partial_stop))
    last_msg = messages[-1].content
    for i in range(len(last_msg) - 1, -1, -1):
        item_type, item_text = last_msg[i].get_type_and_value()
        if item_type == 'text':
            for s in partial_stop:
                if item_text.endswith(s):
                    last_msg[i].text = item_text[:-len(s)]
            break

    return messages


def _truncate_at_stop_word(text: str, stop: List[str]):
    truncated = False
    for s in stop:
        k = text.find(s)
        if k >= 0:
            truncated = True
            text = text[:k]
    return truncated, text


def _postprocess_fncall_messages(messages: List[Message]) -> List[Message]:
    """
    If the model calls function by built-in function call template,
    convert and display it in function_call format.
    """

    # Remove ': ' brought by continued generation of function calling
    last_msg = messages[-1].content
    for i in range(len(last_msg)):
        item_type, item_text = last_msg[i].get_type_and_value()
        if item_type == 'text':
            if item_text.startswith(': '):
                last_msg[i].text = item_text[2:]
            elif item_text.startswith(':'):
                last_msg[i].text = item_text[1:]
            break

    new_messages = []
    for msg in messages:
        role, content = msg.role, msg.content
        assert isinstance(content, list)

        if role in (SYSTEM, USER):
            new_messages.append(Message(role=role, content=content))
            continue

        new_content = []
        for item in content:
            item_type, item_text = item.get_type_and_value()

            if item_type != 'text':  # multimodal
                new_content.append(item)
                continue

            for stop_word in [FN_RESULT, FN_EXIT]:
                assert stop_word in FN_STOP_WORDS
                assert stop_word not in item_text, 'Something wrong, stop words are expected to be excluded.'

            i = item_text.find(f'{FN_NAME}:')

            # If no function call:
            if i < 0:
                show_text = remove_incomplete_special_tokens(item_text)
                if show_text:
                    new_content.append(ContentItem(text=show_text))
                continue

            # If it says something before function call:
            if i > 0:
                answer = item_text[:i].lstrip('\n').rstrip()
                if answer.endswith('\n'):
                    answer = answer[:-1]
                show_text = remove_incomplete_special_tokens(answer)
                if show_text:
                    new_content.append(ContentItem(text=show_text))
                if new_content:
                    new_messages.append(Message(
                        role=role,
                        content=new_content,
                    ))  # split thought and function call
                    new_content = []
                item_text = item_text[i:]

            # If has function call:
            for part in item_text.split(f'{FN_NAME}:'):
                if not part:
                    continue
                if part.endswith('\n'):
                    part = part[:-1]

                arg_sep = f'{FN_ARGS}:'
                i = part.find(arg_sep)
                if i < 0:
                    fn_name = part.strip()
                    list_of_fn_args = ['']
                else:
                    fn_name = part[:i].strip()
                    list_of_fn_args = [_.strip() for _ in part[i + len(arg_sep):].split(arg_sep)]
                fn_name = remove_incomplete_special_tokens(fn_name)
                for fn_args in list_of_fn_args:
                    fn_args = remove_incomplete_special_tokens(fn_args)
                    fn_args = remove_trailing_comment_of_fn_args(fn_args)
                    new_messages.append(
                        Message(
                            role=ASSISTANT,
                            content=[],
                            function_call=FunctionCall(
                                name=fn_name,
                                arguments=fn_args,
                            ),
                        ))
            # Break here and discard the text after function call
            return new_messages

        if new_content:
            new_messages.append(Message(role=role, content=new_content))
    return new_messages


def remove_incomplete_special_tokens(text: str) -> str:
    special_tokens = (FN_NAME, FN_ARGS, FN_RESULT, FN_EXIT)
    text = text.rstrip()
    if text.endswith(special_tokens):
        for s in special_tokens:
            if text.endswith(s):
                text = text[:-len(s)]
                break
    else:
        trail_start = text.rfind('✿')
        trail_token = text[trail_start:]
        for s in special_tokens:
            if s.startswith(trail_token):
                text = text[:trail_start]
                break
    text = text.lstrip('\n').rstrip()
    return text


def remove_trailing_comment_of_fn_args(fn_args: str):
    fn_args = fn_args.strip()

    if fn_args.startswith('{'):
        k = fn_args.rfind('}')
        if k > 0:
            fn_args = fn_args[:k + 1]

    if fn_args.startswith('```'):
        k = fn_args.rfind('\n```')
        if k > 0:
            fn_args = fn_args[:k + 4]

    return fn_args


def _convert_messages_to_target_type(messages: List[Message],
                                     target_type: str) -> Union[List[Message], List[Dict]]:
    if target_type == 'message':
        return [Message(**x) if isinstance(x, dict) else x for x in messages]
    elif target_type == 'dict':
        return [x.model_dump() if not isinstance(x, dict) else x for x in messages]
    else:
        raise NotImplementedError
