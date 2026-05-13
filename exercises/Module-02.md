# Module 2: RAG Pattern Implementation

### 📂 **You'll be editing:** [`src/api/chat.py`](../src/api/chat.py)

### ✅ **Solution:** [`solutions/chat_solution.py`](../solutions/chat_solution.py) — check here if you get stuck

## 📋 Learning Objectives

By the end of this module, you will:

- Understand the Retrieval-Augmented Generation (RAG) pattern
- Build session-based conversation memory
- Write prompt templates for context-aware AI responses
- Format search results as LLM context
- Implement the full RAG pipeline connecting search → LLM → response
- Test your implementation in the running application

## 🎯 What You'll Build

You'll implement the chat backend that powers the application's AI assistant. When a user types a question in the chat panel, your code will:

1. Remember previous messages in the conversation
2. Rephrase follow-up questions into standalone search queries
3. Retrieve relevant listings using vector search (from Module 1)
4. Feed those listings as context to an LLM
5. Return a conversational, grounded response

### Example Conversation:

```
User: "I'm looking for a place in Denver for a weekend getaway"

AI: "I found some great options in Denver! Here are my top recommendations:

1. **Cozy Loft in LoHi** - $125/night
   Perfect for a weekend escape! This modern loft features exposed brick,
   a fully equipped kitchen, and is walking distance to trendy restaurants.

2. **Sunny Studio near RiNo** - $95/night
   Great location for exploring! Close to art galleries and downtown.

Would you like more details about any of these?"

User: "Does the first one have parking?"

AI: "Yes! The Cozy Loft in LoHi includes free street parking..."
```

## 📚 Concept: Retrieval-Augmented Generation (RAG)

### What is RAG?

RAG combines three stages:

1. **Retrieval**: Finding relevant information from a knowledge base (our vector search from Module 1)
2. **Augmentation**: Adding that information to the AI's prompt as context
3. **Generation**: Using an LLM to generate a response grounded in the retrieved context

### Why RAG?

Without RAG:

- ❌ AI only knows what it was trained on (outdated, generic)
- ❌ Can't answer questions about your specific data
- ❌ May "hallucinate" or make up information

With RAG:

- ✅ AI has access to your current, specific data
- ✅ Responses are grounded in real information
- ✅ Can cite sources and provide accurate details
- ✅ Knowledge base is updateable without retraining

### RAG Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         RAG Pipeline                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. User Message                                                 │
│  ┌──────────────────┐                                           │
│  │ "Find cozy       │                                           │
│  │  apartments in   │                                           │
│  │  Denver"         │                                           │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  2. Rephrase (if follow-up)                                     │
│  ┌──────────────────┐                                           │
│  │ Use chat history  │                                           │
│  │ to make question │                                           │
│  │ standalone        │                                           │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  3. Retrieval (Vector Search — from Module 1)                   │
│  ┌──────────────────┐         ┌─────────────────┐              │
│  │ Convert query    │────────▶│ Search DocumentDB│              │
│  │ to embedding     │         │ for similar     │              │
│  └──────────────────┘         │ listings        │              │
│                                └────────┬────────┘              │
│                                         │                        │
│                                         ▼                        │
│  4. Context Formatting                                           │
│  ┌──────────────────────────────────────────┐                  │
│  │ Top 5 listings as readable text:         │                  │
│  │ 1. Loft in LoHi - $125                  │                  │
│  │ 2. Studio near RiNo - $95               │                  │
│  │ ...                                      │                  │
│  └────────┬─────────────────────────────────┘                  │
│           │                                                      │
│           ▼                                                      │
│  5. Augmented Prompt                                             │
│  ┌──────────────────────────────────────────┐                  │
│  │ System: You are a helpful assistant...   │                  │
│  │ Context: [Formatted listings]            │                  │
│  │ User Question: [Original message]        │                  │
│  └────────┬─────────────────────────────────┘                  │
│           │                                                      │
│           ▼                                                      │
│  6. Generation (LLM)                                             │
│  ┌──────────────────┐                                           │
│  │ OpenAI GPT       │──────▶  Conversational response           │
│  └──────────────────┘                                           │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## 🛠️ Step 1: Explore the Existing Code

Before writing anything, take a few minutes to understand how the pieces fit together.

### 1a. Review the search module

Open [`src/api/search.py`](../src/api/search.py) and find:

- **`generate_embedding(text)`** — converts text to a 1536-dimension vector using OpenAI
- **`vector_search(query, limit, filters)`** — runs a `$search` aggregation pipeline against DocumentDB
- **`search_listings(query, limit, filters)`** — the unified search function that tries vector search first, then falls back to text search or static data

These functions are your **retrieval layer**. Your chat module will call `search_listings()` to find relevant listings.

### 1b. Review the API endpoint

Open [`src/api/main.py`](../src/api/main.py) and find the `/query_message` endpoint (around line 220). Notice it calls `generate_chat_response()` — that's the function you'll implement in Step 5.

### 1c. Review the data models

Open [`src/api/models.py`](../src/api/models.py) and look at:

- **`SearchResult`** — has a `listing` (Listing model) and a `score` (similarity score)
- **`ChatRequest`** / **`ChatResponse`** — the API request/response shapes

### 1d. Open the exercise file

Open [`src/api/chat.py`](../src/api/chat.py). This is where you'll work for the rest of Module 2. You'll see:

- Imports and boilerplate (already done)
- `ChatHistory` class (Step 2 — TODO)
- Prompt template strings (Step 3 — TODO)
- `format_listings_for_context()` (Step 4 — TODO)
- `generate_chat_response()` (Step 5 — TODO)

The `get_llm()` function and `format_listings_simple()` helper are already implemented for you.

---

## 🛠️ Step 2: Implement Chat History

📂 **Edit:** [`src/api/chat.py`](../src/api/chat.py) — find the `ChatHistory` class

The `ChatHistory` class stores messages for a single conversation session. Each message is a dictionary with `"role"` (`"user"` or `"assistant"`) and `"content"`.

### Requirements

Implement these methods on the `ChatHistory` class:

| Method                                 | Description                                                                                                                |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `__init__(self, max_messages=20)`      | Initialize an empty message list and store the max                                                                         |
| `add_user_message(self, content)`      | Append `{"role": "user", "content": content}`, then trim                                                                   |
| `add_assistant_message(self, content)` | Append `{"role": "assistant", "content": content}`, then trim                                                              |
| `_trim(self)`                          | If the list exceeds `max_messages`, keep only the last N                                                                   |
| `get_messages(self)`                   | Return a **copy** of the message list                                                                                      |
| `get_formatted_history(self)`          | Return a string like `"User: ...\nAssistant: ..."` for the last 10 messages. Return `"No previous conversation."` if empty |
| `clear(self)`                          | Reset the message list to empty                                                                                            |

### 💡 Why a copy?

`get_messages()` returns `self._messages.copy()` so external code can't accidentally mutate the internal list.

### 💡 Why trim?

Without trimming, conversation history would grow unbounded. Since we include history in prompts, long histories would exceed the LLM's context window and increase cost.

<details>
<summary>🔑 Solution</summary>

```python
class ChatHistory:
    """In-memory chat history manager per session."""

    def __init__(self, max_messages: int = 20):
        self._messages: List[Dict[str, str]] = []
        self._max_messages = max_messages

    def add_user_message(self, content: str):
        self._messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, content: str):
        self._messages.append({"role": "assistant", "content": content})
        self._trim()

    def _trim(self):
        if len(self._messages) > self._max_messages:
            self._messages = self._messages[-self._max_messages:]

    def get_messages(self) -> List[Dict[str, str]]:
        return self._messages.copy()

    def get_formatted_history(self) -> str:
        if not self._messages:
            return "No previous conversation."

        lines = []
        for msg in self._messages[-10:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def clear(self):
        self._messages = []
```

</details>

---

## 🛠️ Step 3: Write the Prompt Templates

📂 **Edit:** [`src/api/chat.py`](../src/api/chat.py) — find the `REPHRASE_PROMPT`, `CONTEXT_PROMPT`, and `FALLBACK_RESPONSE` strings

You need three strings. The first two are **prompt templates** — they contain `{placeholders}` that get filled in at runtime.

### 3a. `REPHRASE_PROMPT`

This prompt takes a conversation history and a follow-up question, and asks the LLM to rephrase the follow-up into a **standalone search query**.

**Why?** When a user says "Does the first one have parking?", the vector search needs a self-contained query like "Does the Denver apartment have parking?" to find relevant results.

**Required placeholders:** `{chat_history}` and `{question}`

### 3b. `CONTEXT_PROMPT`

This is the main system prompt that tells the LLM how to behave. It should include:

- The AI's role (friendly vacation rental assistant)
- Guidelines (only use provided listings, mention price/bedrooms/amenities, be concise, ask follow-up questions)
- The retrieved listings as context
- The user's question

**Required placeholders:** `{context}` and `{question}`

### 3c. `FALLBACK_RESPONSE`

A static string returned when the LLM isn't available (no API key). Tell the user they can still browse listings on the map.

### 💡 Prompt Engineering Tips

- Be specific about what the LLM should and shouldn't do
- "Only recommend listings that appear in the context" prevents hallucination
- "Keep responses concise (under 150 words)" keeps costs and latency down
- "Ask follow-up questions" makes the conversation feel natural

<details>
<summary>🔑 Solution</summary>

```python
REPHRASE_PROMPT = """Given the following conversation history and a follow-up question,
rephrase the follow-up question to be a standalone search query that can be used
to search for vacation rental listings.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Search Query:"""

CONTEXT_PROMPT = """You are a friendly AI assistant helping users find vacation rental listings.
Your responses should be helpful, conversational, and based ONLY on the provided listings.

Guidelines:
- Only recommend listings that appear in the context below
- Mention specific details like price, bedrooms, amenities when relevant
- If no listings match the user's needs, politely say so
- Keep responses concise (under 150 words)
- Be enthusiastic but not pushy
- Ask follow-up questions to better understand user preferences

Available Listings:
{context}

User Question: {question}

Assistant Response:"""

FALLBACK_RESPONSE = """I don't have access to the AI chat features right now.
This could be because:
- OpenAI API key is not configured
- The chat service encountered an error

You can still:
- Browse the listings shown on the map
- Use the search feature to find listings
- Complete Module 2 of the workshop to enable AI-powered chat!

Is there anything else I can help with?"""
```

</details>

---

## 🛠️ Step 4: Format Search Results as Context

📂 **Edit:** [`src/api/chat.py`](../src/api/chat.py) — find `format_listings_for_context()`

This function bridges **retrieval** and **augmentation**. It takes the raw `SearchResult` objects from `search_listings()` and formats them into a human-readable string that gets inserted into `CONTEXT_PROMPT`.

### Input

A list of `SearchResult` objects. Each has:

- `result.listing` — a `Listing` model with fields like `name`, `price`, `property_type`, `bedrooms`, `beds`, `amenities` (list of strings), `description`
- `result.score` — similarity score from vector search (0–1)

### Output

A formatted string like:

```
1. Cozy Loft in LoHi
   Price: $125/night
   Type: Apartment
   Bedrooms: 2 | Beds: 3
   Amenities: Wifi, Kitchen, Heating, Air conditioning, Washer
   Description: Beautiful loft in the heart of LoHi with exposed brick...
   Similarity Score: 0.87

2. Sunny Studio near RiNo
   ...
```

### Requirements

- Return `"No listings available matching the search criteria."` if the list is empty
- Show only the first 5 amenities (avoid overwhelming the LLM)
- Truncate descriptions to 200 characters (add `...` if truncated)
- Handle `None` values gracefully (use `"Not specified"` or `"N/A"`)
- Include the similarity score so the LLM can weight relevance

<details>
<summary>🔑 Solution</summary>

```python
def format_listings_for_context(results: List[SearchResult]) -> str:
    if not results:
        return "No listings available matching the search criteria."

    lines = []
    for i, result in enumerate(results, 1):
        listing = result.listing

        amenities_str = ", ".join(listing.amenities[:5]) if listing.amenities else "Not specified"

        desc = listing.description or ""
        if len(desc) > 200:
            desc = desc[:200] + "..."

        lines.append(f"""
{i}. {listing.name}
   Price: ${listing.price:.0f}/night
   Type: {listing.property_type or 'Not specified'}
   Bedrooms: {listing.bedrooms or 'N/A'} | Beds: {listing.beds or 'N/A'}
   Amenities: {amenities_str}
   Description: {desc}
   Similarity Score: {result.score:.2f}
""")

    return "\n".join(lines)
```

</details>

---

## 🛠️ Step 5: Implement the RAG Pipeline

📂 **Edit:** [`src/api/chat.py`](../src/api/chat.py) — find `generate_chat_response()`

This is the core of Module 2 — the function that ties everything together. When a user sends a chat message, this function:

1. **Gets session history** and the LLM
2. **Records** the user's message in history
3. **Checks** if the LLM is available (return `FALLBACK_RESPONSE` if not)
4. **Rephrases** the question using `REPHRASE_PROMPT` + chat history
5. **Searches** for listings using the rephrased query
6. **Formats** the results as context
7. **Generates** a response using `CONTEXT_PROMPT` + context
8. **Records** the assistant's response in history
9. **Returns** the response text

### Key APIs You'll Use

```python
# Create a prompt template from a string with {placeholders}
prompt = ChatPromptTemplate.from_template(REPHRASE_PROMPT)

# Chain the prompt with the LLM (LangChain Expression Language)
chain = prompt | llm

# Invoke the chain asynchronously (required since this is an async function)
result = await chain.ainvoke({"chat_history": "...", "question": "..."})

# The result is a message object — get the text with:
text = result.content.strip()

# Search for listings (already imported at the top of the function)
results = search_listings(query=search_query, limit=5)
```

### Error Handling

Wrap the main logic in a `try/except`. On failure, fall back to returning plain search results using `format_listings_simple()` — this way the user still gets something useful even if the LLM call fails.

### 💡 Understanding the Rephrase Step

Consider this conversation:

```
User: "Find me a place in Denver with parking"
AI: "Here are 3 options..."
User: "Does the first one have a kitchen?"
```

Without rephrasing, the vector search for "Does the first one have a kitchen?" would find listings about kitchens everywhere — not Denver-specific results.

The rephrase step rewrites it to something like: "Does the Denver apartment listed first have a kitchen?" — giving the search engine much better context.

<details>
<summary>🔑 Solution</summary>

```python
async def generate_chat_response(
    message: str,
    session_id: str = "default"
) -> str:
    from .search import search_listings

    history = get_session_history(session_id)
    llm = get_llm()

    # Add user message to history
    history.add_user_message(message)

    # If LLM not available, return fallback
    if not llm:
        return FALLBACK_RESPONSE

    try:
        # Step 1: Rephrase the question considering chat history
        rephrase_prompt = ChatPromptTemplate.from_template(REPHRASE_PROMPT)
        rephrase_chain = rephrase_prompt | llm

        rephrased = await rephrase_chain.ainvoke({
            "chat_history": history.get_formatted_history(),
            "question": message
        })
        search_query = rephrased.content.strip()

        logger.info(f"Rephrased query: '{message}' -> '{search_query}'")

        # Step 2: Search for relevant listings
        results = search_listings(query=search_query, limit=5)

        # Step 3: Generate response with context
        context = format_listings_for_context(results)

        context_prompt = ChatPromptTemplate.from_template(CONTEXT_PROMPT)
        context_chain = context_prompt | llm

        response = await context_chain.ainvoke({
            "context": context,
            "question": message
        })

        response_text = response.content

        # Add assistant response to history
        history.add_assistant_message(response_text)

        return response_text

    except Exception as e:
        logger.error(f"Chat generation failed: {e}")

        from .search import search_listings
        results = search_listings(query=message, limit=5)

        if results:
            listings_text = format_listings_simple(results)
            response_text = f"I found {len(results)} listings that might interest you:\n\n{listings_text}\n\nWould you like more details about any of these?"
        else:
            response_text = "I couldn't find any listings matching your criteria. Try adjusting your search terms."

        history.add_assistant_message(response_text)
        return response_text
```

</details>

---

## 🧪 Step 6: Test Your Implementation

Now let's verify everything works end-to-end.

### 6a. Restart the API

**Option A (recommended): Refresh the frontend**
Hit `Ctrl + R` on PC or `Cmd + R` on Mac

**Option B: Refresh the API**

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 6b. Test with curl

**Single question:**

```bash
curl -X POST http://localhost:8000/query_message \
  -H "Content-Type: application/json" \
  -d '{"message": "Find me a cozy place in Denver", "session_id": "test1"}'
```

**Follow-up question** (same session_id):

```bash
curl -X POST http://localhost:8000/query_message \
  -H "Content-Type: application/json" \
  -d '{"message": "Does the first one have parking?", "session_id": "test1"}'
```

**Check chat history:**

```bash
curl http://localhost:8000/chat/history?session_id=test1
```

### 6c. Test in the frontend

Open http://localhost:3000 and use the chat panel on the right side. Try a multi-turn conversation:

1. "I'm looking for a place in Denver for a weekend getaway"
2. "Does the first one have parking?"
3. "What about the price? Is it under $150?"

Notice how the AI remembers context from previous messages — that's your conversation memory and rephrasing in action!

### 6d. What to check for

| ✅ Working                              | ❌ Possible Issue                                             |
| --------------------------------------- | ------------------------------------------------------------- |
| AI references specific listings by name | Check `format_listings_for_context()` — context may be empty  |
| Follow-ups reference earlier listings   | Check `REPHRASE_PROMPT` — history may not be passed correctly |
| Prices and details are accurate         | Check that you're reading `listing.price`, not hardcoding     |
| Graceful fallback without API key       | Check `FALLBACK_RESPONSE` is a non-empty string               |

---

## 🎓 What You've Learned

✅ **RAG Pattern**: Combining retrieval, augmentation, and generation  
✅ **Chat History**: Maintaining conversation state across messages  
✅ **Prompt Engineering**: Designing system prompts that constrain and guide the LLM  
✅ **Query Rephrasing**: Rewriting follow-up questions for better retrieval  
✅ **Context Formatting**: Structuring data so the LLM can reason about it  
✅ **End-to-End Integration**: Wiring your code into a running web application

---

## 🚀 Challenges

### Challenge 1: Adjust the Temperature (Easy)

The `get_llm()` function uses `temperature=0.7`. Try changing it and observing the difference:

- `0.0` — Deterministic, factual, repetitive
- `0.7` — Balanced (current)
- `1.0` — More creative, varied

Edit `get_llm()` in `chat.py`, restart the API, and ask the same question multiple times. How do the responses change?

### Challenge 2: Improve the Context Format (Medium)

The current `format_listings_for_context()` shows only 5 amenities and truncates descriptions at 200 characters. Try:

- Showing all amenities
- Including the `neighborhood_overview` field
- Adding the listing URL if available

What happens to response quality? Does longer context always help?

### Challenge 3: Add Sentiment-Aware Responses (Medium)

Modify `CONTEXT_PROMPT` to detect the user's tone and adjust accordingly:

- Frustrated users ("Nothing is working, I just need SOMETHING") → Be extra patient
- Excited users ("OMG this is perfect!") → Match their energy
- Vague users ("Find me something nice") → Ask clarifying questions

### Challenge 4: Extract Filters from Natural Language (Hard)

Before searching, use the LLM to extract structured filters from the user's message:

- "3 bedroom house in Denver under $200 with parking"
- → `{bedrooms: 3, price_max: 200, amenities: ["parking"]}`

Then pass those filters to `search_listings(query=..., filters=extracted_filters)`.

<details>
<summary>💡 Hint</summary>

Create a new prompt template that asks the LLM to return a JSON object with filter fields. Parse the result with `json.loads()`, then pass it to `search_listings()`.

```python
FILTER_EXTRACTION_PROMPT = """Extract search filters from this message.
Return a JSON object with these fields (null if not mentioned):
- bedrooms (number)
- price_max (number)
- property_type (string)
- amenities (array of strings)

Message: {question}

JSON:"""
```

</details>

---

## ✅ Checkpoint

Before moving to Module 3, ensure you have:

- [ ] Implemented the `ChatHistory` class with all methods
- [ ] Written `REPHRASE_PROMPT`, `CONTEXT_PROMPT`, and `FALLBACK_RESPONSE`
- [ ] Implemented `format_listings_for_context()` to format search results
- [ ] Implemented `generate_chat_response()` with the full RAG pipeline
- [ ] Tested single-question chat via the API or frontend
- [ ] Tested multi-turn conversation (follow-up questions work)
- [ ] Completed at least one challenge exercise

## 🎉 What's Next?

In **Module 3: Multi-Agent System with LangGraph**, you'll learn how to:

- Design a multi-agent architecture
- Create specialized agents (Search, Filter, Recommendation)
- Implement agent orchestration with LangGraph
- Handle complex workflows with state management
- Build a conversation router that delegates to the right agent

---

**💬 Stuck?** Compare your code with [`solutions/chat_solution.py`](../solutions/chat_solution.py) or ask your instructor for help!
