# BaseCallbackHandler

> **Class** in `langchain_core`

📖 [View in docs](https://reference.langchain.com/python/langchain-core/callbacks/base/BaseCallbackHandler)

Base callback handler.

## Signature

```python
BaseCallbackHandler()
```

## Extends

- `LLMManagerMixin`
- `ChainManagerMixin`
- `ToolManagerMixin`
- `RetrieverManagerMixin`
- `CallbackManagerMixin`
- `RunManagerMixin`

## Properties

- `raise_error`
- `run_inline`
- `ignore_llm`
- `ignore_retry`
- `ignore_chain`
- `ignore_agent`
- `ignore_retriever`
- `ignore_chat_model`
- `ignore_custom_event`

---

[View source on GitHub](https://github.com/langchain-ai/langchain/blob/04ac76c07ec173a44e3e57de54861d0b637e3299/libs/core/langchain_core/callbacks/base.py#L496)

# RunManagerMixin

> **Class** in `langchain_core`

📖 [View in docs](https://reference.langchain.com/python/langchain-core/callbacks/base/RunManagerMixin)

Mixin for run manager.

## Signature

```python
RunManagerMixin()
```

## Methods

- [`on_text()`](https://reference.langchain.com/python/langchain-core/callbacks/base/RunManagerMixin/on_text)
- [`on_retry()`](https://reference.langchain.com/python/langchain-core/callbacks/base/RunManagerMixin/on_retry)
- [`on_custom_event()`](https://reference.langchain.com/python/langchain-core/callbacks/base/RunManagerMixin/on_custom_event)

---

[View source on GitHub](https://github.com/langchain-ai/langchain/blob/04ac76c07ec173a44e3e57de54861d0b637e3299/libs/core/langchain_core/callbacks/base.py#L435)