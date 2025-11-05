import streamlit as st
from dotenv import load_dotenv
from typing import Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from web_operations import serp_search, reddit_search_api, reddit_post_retrieval
from langchain_google_genai import ChatGoogleGenerativeAI
from prompts import (
    get_reddit_analysis_messages,
    get_google_analysis_messages,
    get_bing_analysis_messages,
    get_reddit_url_analysis_messages,
    get_synthesis_messages
)
import os, time, random

# Load environment variables
load_dotenv()

# ✅ Setup Gemini LLM
llm = ChatGoogleGenerativeAI(
    api_key=os.getenv("google"),
    model="gemini-2.5-flash"
)

# ---------------- SAFE INVOKE ----------------
def safe_invoke(llm, messages, retries=5):
    """Retry wrapper for stable LLM calls"""
    for i in range(retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            wait = (2 ** i) + random.random()
            time.sleep(wait)
    raise RuntimeError("❌ Max retries reached")

# ---------------- STATE ----------------
class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_question: str | None
    google_results: str | None
    bing_results: str | None
    reddit_results: str | None
    selected_reddit_urls: list[str] | None
    reddit_post_data: list | None
    google_analysis: str | None
    bing_analysis: str | None
    reddit_analysis: str | None
    final_answer: str | None

class RedditURLAnalysis(BaseModel):
    selected_urls: List[str] = Field(description="List of relevant Reddit URLs")

# ---------------- GRAPH NODES ----------------
def google_search(state: State):
    q = state["user_question"]
    results = serp_search(q, engine="google")
    return {"google_results": results}

def bing_search(state: State):
    q = state["user_question"]
    results = serp_search(q, engine="bing")
    return {"bing_results": results}

def reddit_search(state: State):
    q = state["user_question"]
    results = reddit_search_api(keyword=q)
    return {"reddit_results": results}

def analyze_reddit_posts(state: State):
    user_question = state["user_question"]
    reddit_results = state.get("reddit_results") or ""
    if not reddit_results:
        return {"selected_reddit_urls": []}

    structured_llm = llm.with_structured_output(RedditURLAnalysis)
    messages = get_reddit_url_analysis_messages(user_question, str(reddit_results))

    try:
        analysis = safe_invoke(structured_llm, messages)
        selected_urls = getattr(analysis, "selected_urls", [])
    except Exception:
        selected_urls = []

    return {"selected_reddit_urls": selected_urls}

def retrieve_reddit_posts(state: State):
    urls = state.get("selected_reddit_urls", [])
    if not urls:
        return {"reddit_post_data": []}
    data = reddit_post_retrieval(urls)
    return {"reddit_post_data": data}

def analyze_google_results(state: State):
    q = state["user_question"]
    google_results = state.get("google_results", "")
    messages = get_google_analysis_messages(q, str(google_results))
    reply = safe_invoke(llm, messages)
    return {"google_analysis": reply.content}

def analyze_bing_results(state: State):
    q = state["user_question"]
    bing_results = state.get("bing_results", "")
    messages = get_bing_analysis_messages(q, str(bing_results))
    reply = safe_invoke(llm, messages)
    return {"bing_analysis": reply.content}

def analyze_reddit_results(state: State):
    q = state["user_question"]
    reddit_results = state.get("reddit_results", "")
    reddit_post_data = state.get("reddit_post_data", [])
    messages = get_reddit_analysis_messages(q, str(reddit_results), reddit_post_data)
    reply = safe_invoke(llm, messages)
    return {"reddit_analysis": reply.content}

def synthesize_analyses(state: State):
    q = state["user_question"]
    messages = get_synthesis_messages(
        q,
        state.get("google_analysis", ""),
        state.get("bing_analysis", ""),
        state.get("reddit_analysis", "")
    )
    reply = safe_invoke(llm, messages)
    return {
        "final_answer": reply.content,
        "messages": [{"role": "assistant", "content": reply.content}],
    }

# ---------------- GRAPH STRUCTURE ----------------
graph_builder = StateGraph(State)
graph_builder.add_node("google_search", google_search)
graph_builder.add_node("bing_search", bing_search)
graph_builder.add_node("reddit_search", reddit_search)
graph_builder.add_node("analyze_reddit_posts", analyze_reddit_posts)
graph_builder.add_node("retrieve_reddit_posts", retrieve_reddit_posts)
graph_builder.add_node("analyze_google_results", analyze_google_results)
graph_builder.add_node("analyze_bing_results", analyze_bing_results)
graph_builder.add_node("analyze_reddit_results", analyze_reddit_results)
graph_builder.add_node("synthesize_analyses", synthesize_analyses)

graph_builder.add_edge(START, "google_search")
graph_builder.add_edge(START, "bing_search")
graph_builder.add_edge(START, "reddit_search")
graph_builder.add_edge("reddit_search", "analyze_reddit_posts")
graph_builder.add_edge("analyze_reddit_posts", "retrieve_reddit_posts")
graph_builder.add_edge("retrieve_reddit_posts", "analyze_google_results")
graph_builder.add_edge("retrieve_reddit_posts", "analyze_bing_results")
graph_builder.add_edge("retrieve_reddit_posts", "analyze_reddit_results")
graph_builder.add_edge("analyze_google_results", "synthesize_analyses")
graph_builder.add_edge("analyze_bing_results", "synthesize_analyses")
graph_builder.add_edge("analyze_reddit_results", "synthesize_analyses")
graph_builder.add_edge("synthesize_analyses", END)

graph = graph_builder.compile()

# ---------------- STREAMLIT UI ----------------
def main():
    st.set_page_config(page_title="Multi-Source Research Agent", page_icon="🔎", layout="wide")
    st.title("🔎 Multi-Source Research Agent")
    st.caption("Google + Bing + Reddit → One summarized answer 💡")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    # Display past messages
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    prompt = st.chat_input("What do you want to research?")
    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare graph state
        state = {
            "messages": [{"role": "user", "content": prompt}],
            "user_question": prompt,
        }

        # Run the graph safely
        with st.spinner("⏳ Running your research agent..."):
            final_state = graph.invoke(state)

        # Display results
        with st.chat_message("assistant"):
            st.markdown(final_state.get("final_answer", "_No answer generated._"))

            st.divider()
            st.subheader("Detailed Analyses")
            tab1, tab2, tab3 = st.tabs(["🌐 Google", "🔍 Bing", "📱 Reddit"])
            with tab1:
                st.markdown(final_state.get("google_analysis", "_No Google analysis._"))
            with tab2:
                st.markdown(final_state.get("bing_analysis", "_No Bing analysis._"))
            with tab3:
                st.markdown(final_state.get("reddit_analysis", "_No Reddit analysis._"))

        st.session_state.chat.append({
            "role": "assistant",
            "content": final_state.get("final_answer", "_No answer generated._"),
        })

if __name__ == "__main__":
    main()
