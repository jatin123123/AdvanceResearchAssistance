# main.py
import os
import time
import random
import operator
from typing import Annotated, List

import streamlit as st
from dotenv import load_dotenv
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI

from web_operations import serp_search, reddit_search_api, reddit_post_retrieval
from prompts import (
    get_reddit_analysis_messages,
    get_google_analysis_messages,
    get_bing_analysis_messages,
    get_reddit_url_analysis_messages,
    get_synthesis_messages,
)

# ---------- Setup ----------
load_dotenv()

llm = ChatGoogleGenerativeAI(
    api_key=os.getenv("google"),
    model="gemini-1.5-flash"
)

# Retry helper (no Streamlit calls)
def safe_invoke(llm, messages, retries=5):
    for i in range(retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            wait = (2 ** i) + random.random()
            time.sleep(wait)
    raise RuntimeError("Max retries reached")

# ---------- State ----------
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
    # Allow concurrent writes; reducer concatenates lists safely
    ui_logs: Annotated[List[str], operator.add]

class RedditURLAnalysis(BaseModel):
    selected_urls: List[str] = Field(
        description="List of Reddit URLs that contain valuable information for answering the user's question"
    )

# ---------- Pure compute nodes (no Streamlit calls) ----------
def google_search(state: State):
    user_question = state.get("user_question", "")
    google_results = serp_search(user_question, engine="google")
    return {
        "google_results": google_results,
        "ui_logs": ["🔍 Searching Google…", "✅ Google search completed"],
    }

def bing_search(state: State):
    user_question = state.get("user_question", "")
    bing_results = serp_search(user_question, engine="bing")
    return {
        "bing_results": bing_results,
        "ui_logs": ["🔍 Searching Bing…", "✅ Bing search completed"],
    }

def reddit_search(state: State):
    user_question = state.get("user_question", "")
    reddit_results = reddit_search_api(keyword=user_question)
    return {
        "reddit_results": reddit_results,
        "ui_logs": ["🔍 Searching Reddit…", "✅ Reddit search completed"],
    }

def analyze_reddit_posts(state: State):
    user_question = state.get("user_question") or ""
    reddit_results = state.get("reddit_results") or ""
    logs = ["🧠 Selecting relevant Reddit posts…"]

    if not reddit_results:
        logs.append("ℹ️ No Reddit results to analyze")
        return {"selected_reddit_urls": [], "ui_logs": logs}

    structured_llm = llm.with_structured_output(RedditURLAnalysis)
    messages = get_reddit_url_analysis_messages(user_question, str(reddit_results))

    try:
        analysis = safe_invoke(structured_llm, messages)
        if hasattr(analysis, "selected_urls"):
            selected_urls = getattr(analysis, "selected_urls", [])
        elif isinstance(analysis, dict) and "selected_urls" in analysis:
            selected_urls = analysis["selected_urls"]
        else:
            selected_urls = []
        logs.append(f"✅ Selected {len(selected_urls)} relevant Reddit posts")
    except Exception as e:
        logs.append(f"⚠️ Error selecting URLs: {e}")
        selected_urls = []

    return {"selected_reddit_urls": selected_urls, "ui_logs": logs}

def retrieve_reddit_posts(state: State):
    logs = ["📥 Retrieving Reddit comments…"]
    selected_urls = state.get("selected_reddit_urls", []) or []
    if not selected_urls:
        logs.append("ℹ️ No Reddit URLs to retrieve")
        return {"reddit_post_data": [], "ui_logs": logs}

    reddit_post_data = reddit_post_retrieval(selected_urls)
    if reddit_post_data:
        logs.append(f"✅ Retrieved comments from {len(selected_urls)} Reddit posts")
    else:
        logs.append("⚠️ Failed to retrieve Reddit post data")
        reddit_post_data = []
    return {"reddit_post_data": reddit_post_data, "ui_logs": logs}

def analyze_google_results(state: State):
    logs = ["🧠 Analyzing Google results…"]
    user_question = state.get("user_question") or ""
    google_results = state.get("google_results") or ""
    messages = get_google_analysis_messages(user_question, str(google_results))
    reply = safe_invoke(llm, messages)
    logs.append("✅ Google analysis completed")
    return {"google_analysis": reply.content, "ui_logs": logs}

def analyze_bing_results(state: State):
    logs = ["🧠 Analyzing Bing results…"]
    user_question = state.get("user_question") or ""
    bing_results = state.get("bing_results") or ""
    messages = get_bing_analysis_messages(user_question, str(bing_results))
    reply = safe_invoke(llm, messages)
    logs.append("✅ Bing analysis completed")
    return {"bing_analysis": reply.content, "ui_logs": logs}

def analyze_reddit_results(state: State):
    logs = ["🧠 Analyzing Reddit results…"]
    user_question = state.get("user_question") or ""
    reddit_results = state.get("reddit_results") or ""
    reddit_post_data = state.get("reddit_post_data") or []
    messages = get_reddit_analysis_messages(user_question, str(reddit_results), reddit_post_data)
    reply = safe_invoke(llm, messages)
    logs.append("✅ Reddit analysis completed")
    return {"reddit_analysis": reply.content, "ui_logs": logs}

def synthesize_analyses(state: State):
    logs = ["🔗 Synthesizing final answer…"]
    user_question = state.get("user_question") or ""
    google_analysis = state.get("google_analysis") or ""
    bing_analysis = state.get("bing_analysis") or ""
    reddit_analysis = state.get("reddit_analysis") or ""
    messages = get_synthesis_messages(user_question, google_analysis, bing_analysis, reddit_analysis)
    reply = safe_invoke(llm, messages)
    final_answer = reply.content
    logs.append("✅ Research synthesis completed!")
    return {
        "final_answer": final_answer,
        "messages": [{"role": "assistant", "content": final_answer}],
        "ui_logs": logs,
    }

# ---------- Graph ----------
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
graph_builder.add_edge("google_search", "analyze_reddit_posts")
graph_builder.add_edge("bing_search", "analyze_reddit_posts")
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

# ---------- Streamlit UI (main thread only) ----------
def main():
    st.set_page_config(
        page_title="Multi-Source Research Agent",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.header("About")
        st.write("Searches Google, Bing, and Reddit, then synthesizes a single answer.")
        st.divider()
        st.subheader("Features")
        st.write("- Parallel multi-source search\n- Intelligent filtering\n- Comprehensive synthesis\n- Real-time status")

    st.title("🔎 Multi-Source Research Agent")
    st.caption("Ask a question, watch the steps, and get a consolidated answer.")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("What would you like to research?", key="research_input")  # unique key
    if not prompt:
        return

    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.status("Running research…", expanded=True) as status:
        status.update(label="Searching sources (Google, Bing, Reddit)…", state="running")

        state: State = {
            "messages": [{"role": "user", "content": prompt}],
            "user_question": prompt,
            "google_results": None,
            "bing_results": None,
            "reddit_results": None,
            "selected_reddit_urls": None,
            "reddit_post_data": None,
            "google_analysis": None,
            "bing_analysis": None,
            "reddit_analysis": None,
            "final_answer": None,
            "ui_logs": [],  # initial list; reducer will merge node outputs
        }

        final_state = graph.invoke(state)

        # Render collected logs after run (all UI updates on main thread)
        for line in final_state.get("ui_logs", []):
            st.write(line)

        status.update(label="Done", state="complete")

    with st.chat_message("assistant"):
        if final_state.get("final_answer"):
            st.markdown(final_state["final_answer"])
        else:
            st.info("No answer was produced. Please try again with a different question.")

        st.divider()
        st.subheader("Detailed Analysis")
        tab1, tab2, tab3 = st.tabs(["🌐 Google", "🔍 Bing", "📱 Reddit"])
        with tab1:
            st.markdown(final_state.get("google_analysis", "_No Google analysis available._"))
        with tab2:
            st.markdown(final_state.get("bing_analysis", "_No Bing analysis available._"))
        with tab3:
            st.markdown(final_state.get("reddit_analysis", "_No Reddit analysis available._"))

        selected_urls = final_state.get("selected_reddit_urls") or []
        if selected_urls:
            with st.expander("Reddit Sources"):
                for i, url in enumerate(selected_urls, 1):
                    st.write(f"{i}. [{url}]({url})")

    st.session_state.chat.append({
        "role": "assistant",
        "content": final_state.get("final_answer", "_No answer produced._"),
    })

if __name__ == "__main__":
    main()
