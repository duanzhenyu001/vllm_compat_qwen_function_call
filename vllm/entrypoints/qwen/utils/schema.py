from typing import List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, field_validator, model_validator

DEFAULT_SYSTEM_MESSAGE = 'You are a helpful assistant.'

ROLE = 'role'
CONTENT = 'content'
NAME = 'name'

SYSTEM = 'system'
USER = 'user'
ASSISTANT = 'assistant'
FUNCTION = 'function'
TOOL = 'tool'

FILE = 'file'
IMAGE = 'image'
OPENAI_DATA_START = "data: "
STREAM_END = "data: [DONE]\n\n"

FN_NAME = '✿FUNCTION✿'
FN_ARGS = '✿ARGS✿'
FN_RESULT = '✿RESULT✿'
FN_EXIT = '✿RETURN✿'
FN_STOP_WORDS = [FN_RESULT, FN_EXIT]

FN_CALL_TEMPLATE_INFO_ZH = """# 工具

## 你拥有如下工具：

{tool_descs}"""

FN_CALL_TEMPLATE_INFO_EN = """# Tools

## You have access to the following tools:

{tool_descs}"""

FN_CALL_TEMPLATE_FMT_ZH = """## 你可以在回复中插入零次、一次或多次以下命令以调用工具：

%s: 工具名称，必须是[{tool_names}]之一。
%s: 工具输入
%s: 工具结果
%s: 根据工具结果进行回复，需将图片用![](url)渲染出来""" % (
    FN_NAME,
    FN_ARGS,
    FN_RESULT,
    FN_EXIT,
)

FN_CALL_TEMPLATE_FMT_EN = """## When you need to call a tool, please insert the following command in your reply, which can be called zero or multiple times according to your needs:

%s: The tool to use, should be one of [{tool_names}]
%s: The input of the tool
%s: Tool results
%s: Reply based on tool results. Images need to be rendered as ![](url)""" % (
    FN_NAME,
    FN_ARGS,
    FN_RESULT,
    FN_EXIT,
)

FN_CALL_TEMPLATE_FMT_PARA_ZH = """## 你可以在回复中插入以下命令以并行调用N个工具：

%s: 工具1的名称，必须是[{tool_names}]之一
%s: 工具1的输入
%s: 工具2的名称
%s: 工具2的输入
...
%s: 工具N的名称
%s: 工具N的输入
%s: 工具1的结果
%s: 工具2的结果
...
%s: 工具N的结果
%s: 根据工具结果进行回复，需将图片用![](url)渲染出来""" % (
    FN_NAME,
    FN_ARGS,
    FN_NAME,
    FN_ARGS,
    FN_NAME,
    FN_ARGS,
    FN_RESULT,
    FN_RESULT,
    FN_RESULT,
    FN_EXIT,
)

FN_CALL_TEMPLATE_FMT_PARA_EN = """## Insert the following command in your reply when you need to call N tools in parallel:

%s: The name of tool 1, should be one of [{tool_names}]
%s: The input of tool 1
%s: The name of tool 2
%s: The input of tool 2
...
%s: The name of tool N
%s: The input of tool N
%s: The result of tool 1
%s: The result of tool 2
...
%s: The result of tool N
%s: Reply based on tool results. Images need to be rendered as ![](url)""" % (
    FN_NAME,
    FN_ARGS,
    FN_NAME,
    FN_ARGS,
    FN_NAME,
    FN_ARGS,
    FN_RESULT,
    FN_RESULT,
    FN_RESULT,
    FN_EXIT,
)

FN_CALL_TEMPLATE = {
    'zh': FN_CALL_TEMPLATE_INFO_ZH + '\n\n' + FN_CALL_TEMPLATE_FMT_ZH,
    'en': FN_CALL_TEMPLATE_INFO_EN + '\n\n' + FN_CALL_TEMPLATE_FMT_EN,
    'zh_parallel': FN_CALL_TEMPLATE_INFO_ZH + '\n\n' + FN_CALL_TEMPLATE_FMT_PARA_ZH,
    'en_parallel': FN_CALL_TEMPLATE_INFO_EN + '\n\n' + FN_CALL_TEMPLATE_FMT_PARA_EN,
}



class BaseModelCompatibleDict(BaseModel):

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def model_dump(self, **kwargs):
        return super().model_dump(exclude_none=True, **kwargs)

    def model_dump_json(self, **kwargs):
        return super().model_dump_json(exclude_none=True, **kwargs)

    def get(self, key, default=None):
        try:
            value = getattr(self, key)
            if value:
                return value
            else:
                return default
        except AttributeError:
            return default

    def __str__(self):
        return f'{self.model_dump()}'


class FunctionCall(BaseModelCompatibleDict):
    name: str
    arguments: str

    def __init__(self, name: str, arguments: str):
        super().__init__(name=name, arguments=arguments)

    def __repr__(self):
        return f'FunctionCall({self.model_dump()})'


class ContentItem(BaseModelCompatibleDict):
    text: Optional[str] = None
    image: Optional[str] = None
    file: Optional[str] = None

    def __init__(self, text: Optional[str] = None, image: Optional[str] = None, file: Optional[str] = None):
        super().__init__(text=text, image=image, file=file)

    @model_validator(mode='after')
    def check_exclusivity(self):
        provided_fields = 0
        if self.text is not None:
            provided_fields += 1
        if self.image:
            provided_fields += 1
        if self.file:
            provided_fields += 1

        if provided_fields != 1:
            raise ValueError("Exactly one of 'text', 'image', or 'file' must be provided.")
        return self

    def __repr__(self):
        return f'ContentItem({self.model_dump()})'

    def get_type_and_value(self) -> Tuple[Literal['text', 'image', 'file'], str]:
        (t, v), = self.model_dump().items()
        assert t in ('text', 'image', 'file')
        return t, v

    @property
    def type(self) -> Literal['text', 'image', 'file']:
        t, v = self.get_type_and_value()
        return t

    @property
    def value(self) -> str:
        t, v = self.get_type_and_value()
        return v


class Message(BaseModelCompatibleDict):
    role: str
    content: Union[str, List[ContentItem]]
    name: Optional[str] = None
    function_call: Optional[FunctionCall] = None

    def __init__(self,
                 role: str,
                 content: Optional[Union[str, List[ContentItem]]],
                 name: Optional[str] = None,
                 function_call: Optional[FunctionCall] = None,
                 **kwargs):
        if content is None:
            content = ''
        super().__init__(role=role, content=content, name=name, function_call=function_call)

    def __repr__(self):
        return f'Message({self.model_dump()})'

    @field_validator('role')
    def role_checker(cls, value: str) -> str:
        if value not in [USER, ASSISTANT, SYSTEM, FUNCTION]:
            raise ValueError(f'{value} must be one of {",".join([USER, ASSISTANT, SYSTEM, FUNCTION])}')
        return value
