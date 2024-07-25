import copy
import json
from typing import Dict

from openai.types.chat import ChatCompletionMessageToolCallParam

from vllm.entrypoints.qwen.settings import DEFAULT_MAX_INPUT_TOKENS, PARALLEL_FUNCTION_CALLS
from vllm.entrypoints.qwen.utils.schema import *
from vllm.entrypoints.qwen.utils.utils import _truncate_input_messages_roughly, has_chinese_messages, \
    format_as_multimodal_message, \
    format_as_text_message, encode_to_qwen_message, std_qwen_to_openai_message, tool_calls_to_functions
from vllm.entrypoints.openai.protocol import (
    ChatCompletionToolsParam,
    ChatCompletionMessageParam,
    ChatCompletionNamedToolChoiceParam
)

from vllm.logger import init_logger

logger = init_logger(__name__)


def pre_process_message(
        messages: List[Union[Dict, ChatCompletionMessageParam]],
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        tools: Optional[List[ChatCompletionToolsParam]] = None,
        tool_choice: Optional[Union[Literal["none", "auto"],
        ChatCompletionNamedToolChoiceParam]] = "none",
) -> List[ChatCompletionMessageParam]:
    for i, msg in enumerate(messages):
        if "tool_calls" in msg and msg["tool_calls"]:
            msg["tool_calls"] = [ChatCompletionMessageToolCallParam(**t) for t in msg["tool_calls"]]
    messages = copy.deepcopy(messages)
    logger.info(f"before encode: {messages}")
    messages = encode_to_qwen_message(messages)
    logger.info(f"after qwen encode: {messages}")
    if messages[0].role != SYSTEM:
        messages = [Message(role=SYSTEM, content=DEFAULT_SYSTEM_MESSAGE)] + messages
    # Not precise. It's hard to estimate tokens related with function calling and multimodal items.
    if max_input_tokens > 0:
        messages = _truncate_input_messages_roughly(
            messages=messages,
            max_tokens=max_input_tokens
        )

    lang: Literal['en', 'zh'] = 'zh' if has_chinese_messages(messages) else 'en'

    logger.info(f"language: {lang}")

    functions = None
    if tools:
        functions = tool_calls_to_functions(tools)
    fn_choice = tool_choice if isinstance(tool_choice, str) else tool_choice.function.name
    if functions:
        fncall_mode = True
    else:
        fncall_mode = False
    if fn_choice not in ["none", "auto"]:
        valid_fn_choices = [f.get('name', f.get('name_for_model', None)) for f in (functions or [])]
        valid_fn_choices = [f for f in valid_fn_choices if f]
        if fn_choice not in valid_fn_choices:
            raise ValueError(f'The value of function_choice must be one of the following: {valid_fn_choices}. '
                             f'But function_choice="{fn_choice}" is received.')
        if fn_choice == 'none':
            fncall_mode = False

    messages = [format_as_multimodal_message(msg, add_upload_info=True, lang=lang) for msg in messages]
    if not fncall_mode:
        messages = _remove_fncall_messages(messages, lang=lang)
    else:
        messages = _preprocess_fncall_messages(messages)

    parallel_function_calls = PARALLEL_FUNCTION_CALLS
    messages = _prepend_fncall_system(
        messages=messages,
        functions=functions,
        lang=lang,
        parallel_function_calls=parallel_function_calls,
    )

    if fn_choice not in ('auto', 'none'):
        if messages[-1].role == ASSISTANT:
            msg_to_cont = copy.deepcopy(messages[-1])
            if msg_to_cont.content.endswith(FN_EXIT):
                msg_to_cont.content += ': '
            msg_to_cont.content += '\n'
            messages = messages[:-1]
        else:
            msg_to_cont = Message(role=ASSISTANT, content='')
        msg_to_cont.content += f'{FN_NAME}: {fn_choice}'
        messages = messages + [msg_to_cont]
    messages = combine_latest_assistant_to_user(messages)
    messages = std_qwen_to_openai_message(messages)
    return messages


def combine_latest_assistant_to_user(messages: List[Message]) -> List[Message]:
    if messages and messages[-1].role == ASSISTANT:
        assert len(messages) > 1 and messages[-2].role == USER
        assert messages[-1].function_call is None
        usr = messages[-2].content
        bot = messages[-1].content
        sep = '\n\n'
        if isinstance(usr, str) and isinstance(bot, str):
            usr = usr + sep + bot
        elif isinstance(usr, list) and isinstance(bot, list):
            usr = usr + [ContentItem(text=sep)] + bot
        else:
            raise NotImplementedError
        text_to_complete = copy.deepcopy(messages[-2])
        text_to_complete.content = usr
        messages = messages[:-2] + [text_to_complete]
    return messages


def _prepend_fncall_system(
        messages: List[Message],
        functions: List[Dict],
        lang: Literal['en', 'zh'],
        parallel_function_calls: bool = False,
) -> List[Message]:
    tool_desc_template = FN_CALL_TEMPLATE[lang + ('_parallel' if parallel_function_calls else '')]
    tool_descs = '\n\n'.join(get_function_description(function, lang=lang) for function in functions)
    tool_names = ','.join(function.get('name', function.get('name_for_model', '')) for function in functions)
    tool_system = tool_desc_template.format(tool_descs=tool_descs, tool_names=tool_names)

    assert messages[0].role == SYSTEM
    messages = copy.deepcopy(messages[:1]) + messages[1:]
    if isinstance(messages[0].content, str):
        messages[0].content += '\n\n' + tool_system
    else:
        messages[0].content.append(ContentItem(text='\n\n' + tool_system))

    return messages


def get_function_description(function: Dict, lang: Literal['en', 'zh']) -> str:
    """
    Text description of function
    """
    tool_desc_template = {
        'zh': '### {name_for_human}\n\n{name_for_model}: {description_for_model} 输入参数：{parameters} {args_format}',
        'en': '### {name_for_human}\n\n{name_for_model}: {description_for_model} Parameters: {parameters} {args_format}'
    }
    tool_desc = tool_desc_template[lang]
    name = function.get('name', None)
    name_for_human = function.get('name_for_human', name)
    name_for_model = function.get('name_for_model', name)
    assert name_for_human and name_for_model

    if name_for_model == 'code_interpreter':
        args_format = {
            'zh': '此工具的输入应为Markdown代码块。',
            'en': 'Enclose the code within triple backticks (`) at the beginning and end of the code.',
        }
    else:
        args_format = {
            'zh': '此工具的输入应为JSON对象。',
            'en': 'Format the arguments as a JSON object.',
        }
    args_format = function.get('args_format', args_format[lang])

    return tool_desc.format(name_for_human=name_for_human,
                            name_for_model=name_for_model,
                            description_for_model=function['description'],
                            parameters=json.dumps(function['parameters'], ensure_ascii=False),
                            args_format=args_format).rstrip()


def _remove_fncall_messages(messages: List[Message], lang: Literal['en', 'zh']) -> List[Message]:
    # Change function calls into user messages so that the model won't try
    # to generate function calls when given functions and function_choice="none".
    new_messages = []
    for msg in messages:
        if (msg.role == FUNCTION) or msg.function_call:
            if (not new_messages) or (new_messages[-1].role != USER):
                new_messages.append(Message(role=USER, content=[]))
            if msg.function_call:
                tool_name = msg.function_call.name
                tool_args = msg.function_call.arguments
                if lang == 'zh':
                    tool_text = f'\n\n工具"{tool_name}"被调用时使用了以下参数：\n{tool_args}'
                else:
                    tool_text = f'\n\nThe tool "{tool_name}" was called with these arguments:\n{tool_args}'
            else:
                assert msg.role == FUNCTION
                if msg.content:
                    assert len(msg.content) == 1
                    assert isinstance(msg.content[0], ContentItem)
                    assert isinstance(msg.content[0].text, str)
                    tool_result = msg.content[0].text
                else:
                    tool_result = 'No result.'
                if lang == 'zh':
                    tool_text = f'\n\n该工具返回了以下结果：\n{tool_result}'
                else:
                    tool_text = f'\n\nThe tool has returned the following result:\n{tool_result}'
            new_messages[-1].content.append(ContentItem(text=tool_text))
        else:
            if (msg.role == USER) and new_messages and (new_messages[-1].role == USER):
                # Separate two user messages with an assistant message to make the bot focus on the latter:
                new_messages.append(Message(role=ASSISTANT, content=[ContentItem(text='...')]))
            new_messages.append(msg)
    return new_messages


def _preprocess_fncall_messages(messages: List[Message]) -> List[Message]:
    """Convert messages with function_call key and function role to assistant's content, which is
        for chat interface or text_completion interface that do not support functions.
    """
    validate_num_fncall_results(messages)
    new_messages = []
    for msg in copy.deepcopy(messages):
        role, content = msg.role, msg.content
        if role in (SYSTEM, USER):
            new_messages.append(msg)
        elif role == ASSISTANT:
            content = (content or [])
            fn_call = msg.function_call
            if fn_call:
                f_name = fn_call.name
                f_args = fn_call.arguments
                if f_args.startswith('```'):  # if code snippet
                    f_args = '\n' + f_args  # for markdown rendering
                func_content = '\n' if new_messages[-1].role == ASSISTANT else ''
                func_content += f'{FN_NAME}: {f_name}'
                func_content += f'\n{FN_ARGS}: {f_args}'
                content.append(ContentItem(text=func_content))
            if new_messages[-1].role == ASSISTANT:
                new_messages[-1].content += content
            else:
                new_messages.append(Message(role=role, content=content))
        elif role == FUNCTION:
            assert new_messages[-1].role == ASSISTANT
            assert isinstance(content, list)
            if content:
                assert len(content) == 1
                assert isinstance(content[0], ContentItem)
                f_result = content[0].text
                assert f_result is not None
            else:
                f_result = ''
            f_exit = f'\n{FN_EXIT}: '
            last_text_content = new_messages[-1].content[-1].text
            if last_text_content.endswith(f_exit):
                new_messages[-1].content[-1].text = last_text_content[:-len(f_exit)]
            new_messages[-1].content += [ContentItem(text=f'\n{FN_RESULT}: {f_result}{f_exit}')]
        else:
            raise TypeError

    # Remove ': ' for continued generation of function calling,
    # because ': ' may form a single token with its following words
    if new_messages[-1].role == ASSISTANT:
        last_msg = new_messages[-1].content
        for i in range(len(last_msg) - 1, -1, -1):
            item_type, item_text = last_msg[i].get_type_and_value()
            if item_type == 'text':
                if item_text.endswith(f'{FN_EXIT}: '):
                    last_msg[i].text = item_text[:-2]
                break
    new_messages = [format_as_text_message(msg, add_upload_info=False) for msg in new_messages]
    return new_messages


def validate_num_fncall_results(messages: List[Message]):
    fn_results = []
    i = len(messages) - 1
    while messages[i].role == FUNCTION:
        fn_results = [messages[i].name] + fn_results
        i -= 1

    fn_calls = []
    while messages[i].function_call:
        fn_calls = [messages[i].function_call.name] + fn_calls
        i -= 1

    if len(fn_calls) != len(fn_results):
        raise ValueError(f'Expecting {len(fn_calls)} function results (i.e., messages with role="function") '
                         f'but received {len(fn_results)} function results. '
                         'The number of function results must match that of the function_call messages.')
    for fc_name, fr_name in zip(fn_calls, fn_results):
        if fr_name and (fc_name != fr_name):
            raise ValueError('The function results (i.e., the messages with role="function" ) must be '
                             'put in the same order as the function_call messages. And the function names must match.'
                             f'The function results are currently {fn_results}. But {fn_calls} are expected.')


def exist_tool_data(messages: List[Dict]) -> bool:
    for msg in messages:
        if "role" in msg:
            if msg["role"] == ASSISTANT:
                if "tool_calls" in msg:
                    return True
            if msg["role"] == TOOL:
                return True
    return False
