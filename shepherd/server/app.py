import json
import multiprocessing
from re import compile, Match, Pattern
from threading import Lock
from functools import partial
from typing import Callable, Coroutine, Iterator, List, Optional, Tuple, Union, Dict
from typing_extensions import TypedDict, Literal

import llama2c_py

import anyio
from anyio.streams.memory import MemoryObjectSendStream
from starlette.concurrency import run_in_threadpool, iterate_in_threadpool
from fastapi import Depends, FastAPI, APIRouter, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from sse_starlette.sse import EventSourceResponse

from typing import Any, List, Optional, Dict, Union
from typing_extensions import TypedDict, NotRequired, Literal


class Settings(BaseSettings):
    model: str = Field(
        description="The path to the model to use for generating completions."
    )
    model_alias: Optional[str] = Field(
        default=None,
        description="The alias of the model to use for generating completions.",
    )
    n_ctx: int = Field(default=2048, ge=1, description="The context size.")
    n_gpu_layers: int = Field(
        default=0,
        ge=0,
        description="The number of layers to put on the GPU. The rest will be on the CPU.",
    )
    tensor_split: Optional[List[float]] = Field(
        default=None,
        description="Split layers across multiple GPUs in proportion.",
    )
    rope_freq_base: float = Field(default=10000, ge=1, description="RoPE base frequency")
    rope_freq_scale: float = Field(default=1.0, description="RoPE frequency scaling factor")
    seed: int = Field(
        default=1337, description="Random seed. -1 for random."
    )
    n_batch: int = Field(
        default=512, ge=1, description="The batch size to use per eval."
    )
    n_threads: int = Field(
        default=max(multiprocessing.cpu_count() // 2, 1),
        ge=1,
        description="The number of threads to use.",
    )
    f16_kv: bool = Field(default=True, description="Whether to use f16 key/value.")
    """
    use_mlock: bool = Field(
        default=llama2c_py.llama_mlock_supported(),
        description="Use mlock.",
    )
    use_mmap: bool = Field(
        default=llama2c_py.llama_mmap_supported(),
        description="Use mmap.",
    )
    """
    embedding: bool = Field(default=True, description="Whether to use embeddings.")
    low_vram: bool = Field(
        default=False,
        description="Whether to use less VRAM. This will reduce performance.",
    )
    last_n_tokens_size: int = Field(
        default=64,
        ge=0,
        description="Last n tokens to keep for repeat penalty calculation.",
    )
    logits_all: bool = Field(default=True, description="Whether to return logits.")
    cache: bool = Field(
        default=False,
        description="Use a cache to reduce processing times for evaluated prompts.",
    )
    cache_type: Literal["ram", "disk"] = Field(
        default="ram",
        description="The type of cache to use. Only used if cache is True.",
    )
    cache_size: int = Field(
        default=2 << 30,
        description="The size of the cache in bytes. Only used if cache is True.",
    )
    vocab_only: bool = Field(
        default=False, description="Whether to only return the vocabulary."
    )
    verbose: bool = Field(
        default=True, description="Whether to print debug information."
    )
    host: str = Field(default="localhost", description="Listen address")
    port: int = Field(default=8000, description="Listen port")
    interrupt_requests: bool = Field(
        default=True,
        description="Whether to interrupt requests when a new request is received.",
    )
    n_gqa: Optional[int] = Field(
        default=None,
        description="TEMPORARY: Set to 8 for Llama2 70B",
    )
    rms_norm_eps: Optional[float] = Field(
        default=None,
        description="TEMPORARY",
    )
    mul_mat_q: Optional[bool] = Field(
        default=None,
        description="TEMPORARY",
    )


class ErrorResponse(TypedDict):
    """OpenAI style error response"""

    message: str
    type: str
    param: Optional[str]
    code: Optional[str]



class ErrorResponseFormatters:
    """Collection of formatters for error responses.

    Args:
        request (Union[CreateCompletionRequest, CreateChatCompletionRequest]):
            Request body
        match (Match[str]): Match object from regex pattern

    Returns:
        Tuple[int, ErrorResponse]: Status code and error response
    """

    @staticmethod
    def context_length_exceeded(
        request: Union[
            "CreateCompletionRequest", "CreateChatCompletionRequest"
        ],
        match, # type: Match[str] # type: ignore
    ) -> Tuple[int, ErrorResponse]:
        """Formatter for context length exceeded error"""

        context_window = int(match.group(2))
        prompt_tokens = int(match.group(1))
        completion_tokens = request.max_tokens
        if hasattr(request, "messages"):
            # Chat completion
            message = (
                "This model's maximum context length is {} tokens. "
                "However, you requested {} tokens "
                "({} in the messages, {} in the completion). "
                "Please reduce the length of the messages or completion."
            )
        else:
            # Text completion
            message = (
                "This model's maximum context length is {} tokens, "
                "however you requested {} tokens "
                "({} in your prompt; {} for the completion). "
                "Please reduce your prompt; or completion length."
            )
        return 400, ErrorResponse(
            message=message.format(
                context_window,
                completion_tokens + prompt_tokens,
                prompt_tokens,
                completion_tokens,
            ),
            type="invalid_request_error",
            param="messages",
            code="context_length_exceeded",
        )

    @staticmethod
    def model_not_found(
        request: Union[
            "CreateCompletionRequest", "CreateChatCompletionRequest"
        ],
        match # type: Match[str] # type: ignore
    ) -> Tuple[int, ErrorResponse]:
        """Formatter for model_not_found error"""

        model_path = str(match.group(1))
        message = f"The model `{model_path}` does not exist"
        return 400, ErrorResponse(
            message=message,
            type="invalid_request_error",
            param=None,
            code="model_not_found",
        )



class RouteErrorHandler(APIRoute):
    """Custom APIRoute that handles application errors and exceptions"""

    # key: regex pattern for original error message from llama_cpp
    # value: formatter function
    pattern_and_formatters: Dict[
        "Pattern",
        Callable[
            [
                Union["CreateCompletionRequest", "CreateChatCompletionRequest"],
                "Match[str]",
            ],
            Tuple[int, ErrorResponse],
        ],
    ] = {
        compile(
            r"Requested tokens \((\d+)\) exceed context window of (\d+)"
        ): ErrorResponseFormatters.context_length_exceeded,
        compile(
            r"Model path does not exist: (.+)"
        ): ErrorResponseFormatters.model_not_found,
    }

    def error_message_wrapper(
        self,
        error: Exception,
        body: Optional[
            Union[
                "CreateChatCompletionRequest",
                "CreateCompletionRequest",
                "CreateEmbeddingRequest",
            ]
        ] = None,
    ) -> Tuple[int, ErrorResponse]:
        """Wraps error message in OpenAI style error response"""

        if body is not None and isinstance(
            body,
            (
                CreateCompletionRequest,
                CreateChatCompletionRequest,
            ),
        ):
            # When text completion or chat completion
            for pattern, callback in self.pattern_and_formatters.items():
                match = pattern.search(str(error))
                if match is not None:
                    return callback(body, match)

        # Wrap other errors as internal server error
        return 500, ErrorResponse(
            message=str(error),
            type="internal_server_error",
            param=None,
            code=None,
        )

    def get_route_handler(
        self,
    ) -> Callable[[Request], Coroutine[None, None, Response]]:
        """Defines custom route handler that catches exceptions and formats
        in OpenAI style error response"""

        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            try:
                return await original_route_handler(request)
            except Exception as exc:
                json_body = await request.json()
                try:
                    if "messages" in json_body:
                        # Chat completion
                        body: Optional[
                            Union[
                                CreateChatCompletionRequest,
                                CreateCompletionRequest,
                                CreateEmbeddingRequest,
                            ]
                        ] = CreateChatCompletionRequest(**json_body)
                    elif "prompt" in json_body:
                        # Text completion
                        body = CreateCompletionRequest(**json_body)
                    else:
                        # Embedding
                        body = CreateEmbeddingRequest(**json_body)
                except Exception:
                    # Invalid request body
                    body = None

                # Get proper error message from the exception
                (
                    status_code,
                    error_message,
                ) = self.error_message_wrapper(error=exc, body=body)
                return JSONResponse(
                    {"error": error_message},
                    status_code=status_code,
                )

        return custom_route_handler



router = APIRouter(route_class=RouteErrorHandler)

class ModelData(TypedDict):
    id: str
    object: Literal["model"]
    owned_by: str
    permissions: List[str]



class ModelList(TypedDict):
    object: Literal["list"]
    data: List[ModelData]


@router.get("/v1/models")
async def get_models() -> ModelList:
    #assert llama is not None
    print("__func__") 
    return {
        "object": "list",
        "data": [
            {
                "id": "12321",
                "object": "model",
                "owned_by": "me",
                "permissions": [],
            }
        ],
    }

@router.get("/")
async def root():
    return {"message": "Hello World"}


model_field = Field(description="The model to use for generating completions.", default=None)

max_tokens_field = Field(
    default=16, ge=1, description="The maximum number of tokens to generate."
)

temperature_field = Field(
    default=0.8,
    ge=0.0,
    le=2.0,
    description="Adjust the randomness of the generated text.\n\n"
    + "Temperature is a hyperparameter that controls the randomness of the generated text. It affects the probability distribution of the model's output tokens. A higher temperature (e.g., 1.5) makes the output more random and creative, while a lower temperature (e.g., 0.5) makes the output more focused, deterministic, and conservative. The default value is 0.8, which provides a balance between randomness and determinism. At the extreme, a temperature of 0 will always pick the most likely next token, leading to identical outputs in each run.",
)

top_p_field = Field(
    default=0.95,
    ge=0.0,
    le=1.0,
    description="Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P.\n\n"
    + "Top-p sampling, also known as nucleus sampling, is another text generation method that selects the next token from a subset of tokens that together have a cumulative probability of at least p. This method provides a balance between diversity and quality by considering both the probabilities of tokens and the number of tokens to sample from. A higher value for top_p (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text.",
)

stop_field = Field(
    default=None,
    description="A list of tokens at which to stop generation. If None, no stop tokens are used.",
)

stream_field = Field(
    default=True,
    description="Whether to stream the results as they are generated. Useful for chatbots.",
)

top_k_field = Field(
    default=40,
    ge=0,
    description="Limit the next token selection to the K most probable tokens.\n\n"
    + "Top-k sampling is a text generation method that selects the next token only from the top k most likely tokens predicted by the model. It helps reduce the risk of generating low-probability or nonsensical tokens, but it may also limit the diversity of the output. A higher value for top_k (e.g., 100) will consider more tokens and lead to more diverse text, while a lower value (e.g., 10) will focus on the most probable tokens and generate more conservative text.",
)

repeat_penalty_field = Field(
    default=1.1,
    ge=0.0,
    description="A penalty applied to each token that is already generated. This helps prevent the model from repeating itself.\n\n"
    + "Repeat penalty is a hyperparameter used to penalize the repetition of token sequences during text generation. It helps prevent the model from generating repetitive or monotonous text. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient.",
)

presence_penalty_field = Field(
    default=0.0,
    ge=-2.0,
    le=2.0,
    description="Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.",
)

frequency_penalty_field = Field(
    default=0.0,
    ge=-2.0,
    le=2.0,
    description="Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.",
)

mirostat_mode_field = Field(
    default=0,
    ge=0,
    le=2,
    description="Enable Mirostat constant-perplexity algorithm of the specified version (1 or 2; 0 = disabled)",
)

mirostat_tau_field = Field(
    default=5.0,
    ge=0.0,
    le=10.0,
    description="Mirostat target entropy, i.e. the target perplexity - lower values produce focused and coherent text, larger values produce more diverse and less coherent text",
)

mirostat_eta_field = Field(
    default=0.1, ge=0.001, le=1.0, description="Mirostat learning rate"
)



class CreateCompletionRequest(BaseModel):
    prompt: Union[str, List[str]] = Field(
        default="", description="The prompt to generate completions for."
    )
    suffix: Optional[str] = Field(
        default=None,
        description="A suffix to append to the generated text. If None, no suffix is appended. Useful for chatbots.",
    )
    max_tokens: int = max_tokens_field
    temperature: float = temperature_field
    top_p: float = top_p_field
    mirostat_mode: int = mirostat_mode_field
    mirostat_tau: float = mirostat_tau_field
    mirostat_eta: float = mirostat_eta_field
    echo: bool = Field(
        default=False,
        description="Whether to echo the prompt in the generated text. Useful for chatbots.",
    )
    stop: Optional[Union[str, List[str]]] = stop_field
    stream: bool = stream_field
    logprobs: Optional[int] = Field(
        default=None,
        ge=0,
        description="The number of logprobs to generate. If None, no logprobs are generated.",
    )
    presence_penalty: Optional[float] = presence_penalty_field
    frequency_penalty: Optional[float] = frequency_penalty_field
    logit_bias: Optional[Dict[str, float]] = Field(None)
    logprobs: Optional[int] = Field(None)

    # ignored or currently unsupported
    model: Optional[str] = model_field
    n: Optional[int] = 1
    best_of: Optional[int] = 1
    user: Optional[str] = Field(default=None)

    # llama.cpp specific parameters
    top_k: int = top_k_field
    repeat_penalty: float = repeat_penalty_field
    logit_bias_type: Optional[Literal["input_ids", "tokens"]] = Field(None)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "\n\n### Instructions:\nWhat is the capital of France?\n\n### Response:\n",
                    "stop": ["\n", "###"],
                }
            ]
        }
    }



class EmbeddingUsage(TypedDict):
    prompt_tokens: int
    total_tokens: int


class EmbeddingData(TypedDict):
    index: int
    object: str
    embedding: List[float]


class Embedding(TypedDict):
    object: Literal["list"]
    model: str
    data: List[EmbeddingData]
    usage: EmbeddingUsage


class CompletionLogprobs(TypedDict):
    text_offset: List[int]
    token_logprobs: List[Optional[float]]
    tokens: List[str]
    top_logprobs: List[Optional[Dict[str, float]]]


class CompletionChoice(TypedDict):
    text: str
    index: int
    logprobs: Optional[CompletionLogprobs]
    finish_reason: Optional[str]


class CompletionUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionChunk(TypedDict):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChoice]



class Completion(TypedDict):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage


@router.post(
    "/v1/completions",
)
async def create_completion(
    request: Request,
    body: CreateCompletionRequest
) -> str:
    if isinstance(body.prompt, list):
        assert len(body.prompt) <= 1
        body.prompt = body.prompt[0] if len(body.prompt) > 0 else ""

    exclude = {
        "n",
        "logit_bias",
        "logit_bias_type",
        "user",
    }


    kwargs = body.model_dump(exclude=exclude)

    print("__func__: kwargs = ", kwargs)

    ret_string = await run_in_threadpool(inference, "stories15M.bin")
    print("ret_string: ", ret_string)
    return ret_string
    return ret_string

    if isinstance(iterator_or_completion, Iterator):
        print("__func__: iterator_or_completion = Iterator")
        # EAFP: It's easier to ask for forgiveness than permission
        first_response = await run_in_threadpool(next, iterator_or_completion)

        # If no exception was raised from first_response, we can assume that
        # the iterator is valid and we can use it to stream the response.
        def iterator() -> Iterator[llama_cpp.CompletionChunk]:
            yield first_response
            yield from iterator_or_completion

        send_chan, recv_chan = anyio.create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan, data_sender_callable=partial(  # type: ignore
                get_event_publisher,
                request=request,
                inner_send_chan=send_chan,
                iterator=iterator(),
            )
        )
    else:
        print("iterator_or_completion is not Iterator= ", iterator_or_completion)
        return iterator_or_completion


settings: Optional[Settings] = None
llama = None #: Optional[llama2c_py.inference] = None

from llama2c_py.llama2c_py import *

def create_app(settings: Optional[Settings] = None):
    if settings is None:
        settings = Settings()
    app = FastAPI(
        title="🦙 llama.cpp Python API",
        version="0.0.1",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    global llama
    print("setting ===============")
    print(settings)
    # para 0 is argc, remove later
    print("model_path=", settings.model)
    #llama = llama2c_py.llama2c_py.inference(2, model_path=settings.model)

    return app


