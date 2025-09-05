import streamlit as st
from dotenv import load_dotenv
from typing import Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
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
import os
import time
import random

load_dotenv()

# ✅ Setup Gemini LLM
llm = ChatGoogleGenerativeAI(
    api_key=os.getenv("google"),
    model="gemini-1.5-flash"   # ⚡ more quota than gemini-2.5-pro
)

# ✅ Retry handler
def safe_invoke(llm, messages, retries=5):
    for i in range(retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            wait = (2 ** i) + random.random()
            st.write(f"⚠️ Retry {i+1}/{retries}, waiting {wait:.1f}s due to: {e}")
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
    selected_urls: List[str] = Field(
        description="List of Reddit URLs that contain valuable information for answering the user's question"
    )

# ---------------- NODES ----------------
def google_search(state: State):
    user_question = state.get("user_question", "")
    st.markdown('<div class="step-indicator">🔍 Searching Google...</div>', unsafe_allow_html=True)
    google_results = serp_search(user_question, engine="google")
    st.success("✅ Google search completed")
    return {"google_results": google_results}

def bing_search(state: State):
    user_question = state.get("user_question", "")
    st.markdown('<div class="step-indicator">🔍 Searching Bing...</div>', unsafe_allow_html=True)
    bing_results = serp_search(user_question, engine="bing")
    st.success("✅ Bing search completed")
    return {"bing_results": bing_results}

def reddit_search(state: State):
    user_question = state.get("user_question", "")
    st.markdown('<div class="step-indicator">🔍 Searching Reddit...</div>', unsafe_allow_html=True)
    reddit_results = reddit_search_api(keyword=user_question)
    st.success("✅ Reddit search completed")
    return {"reddit_results": reddit_results}

def analyze_reddit_posts(state: State):
    st.markdown('<div class="step-indicator">🧠 Analyzing Reddit posts...</div>', unsafe_allow_html=True)
    user_question = state.get("user_question") or ""
    reddit_results = state.get("reddit_results") or ""

    if not reddit_results:
        st.info("ℹ️ No Reddit results to analyze")
        return {"selected_reddit_urls": []}

    structured_llm = llm.with_structured_output(RedditURLAnalysis)
    messages = get_reddit_url_analysis_messages(user_question, str(reddit_results))

    try:
        analysis = safe_invoke(structured_llm, messages)
        # Handle both dictionary and BaseModel responses
        if hasattr(analysis, 'selected_urls'):
            selected_urls = getattr(analysis, 'selected_urls', [])
        elif isinstance(analysis, dict) and 'selected_urls' in analysis:
            selected_urls = analysis['selected_urls']
        else:
            selected_urls = []
            
        st.success(f"✅ Selected {len(selected_urls)} relevant Reddit posts")
        if selected_urls:
            with st.expander("🔗 View Selected URLs"):
                for i, url in enumerate(selected_urls, 1):
                    st.write(f"{i}. {url}")
    except Exception as e:
        st.error(f"⚠️ Error selecting URLs: {e}")
        selected_urls = []

    return {"selected_reddit_urls": selected_urls}

def retrieve_reddit_posts(state: State):
    st.markdown('<div class="step-indicator">📥 Retrieving Reddit comments...</div>', unsafe_allow_html=True)
    selected_urls = state.get("selected_reddit_urls", [])
    if not selected_urls:
        st.info("ℹ️ No Reddit URLs to retrieve")
        return {"reddit_post_data": []}

    reddit_post_data = reddit_post_retrieval(selected_urls)
    if reddit_post_data:
        st.success(f"✅ Retrieved comments from {len(selected_urls)} Reddit posts")
    else:
        st.warning("⚠️ Failed to retrieve Reddit post data")
        reddit_post_data = []
    return {"reddit_post_data": reddit_post_data}

def analyze_google_results(state: State):
    st.markdown('<div class="step-indicator">🧠 Analyzing Google results...</div>', unsafe_allow_html=True)
    user_question = state.get("user_question") or ""
    google_results = state.get("google_results") or ""
    messages = get_google_analysis_messages(user_question, str(google_results))
    reply = safe_invoke(llm, messages)
    st.success("✅ Google analysis completed")
    return {"google_analysis": reply.content}

def analyze_bing_results(state: State):
    st.markdown('<div class="step-indicator">🧠 Analyzing Bing results...</div>', unsafe_allow_html=True)
    user_question = state.get("user_question") or ""
    bing_results = state.get("bing_results") or ""
    messages = get_bing_analysis_messages(user_question, str(bing_results))
    reply = safe_invoke(llm, messages)
    st.success("✅ Bing analysis completed")
    return {"bing_analysis": reply.content}

def analyze_reddit_results(state: State):
    st.markdown('<div class="step-indicator">🧠 Analyzing Reddit results...</div>', unsafe_allow_html=True)
    user_question = state.get("user_question") or ""
    reddit_results = state.get("reddit_results") or ""
    reddit_post_data = state.get("reddit_post_data") or []
    messages = get_reddit_analysis_messages(user_question, str(reddit_results), reddit_post_data)
    reply = safe_invoke(llm, messages)
    st.success("✅ Reddit analysis completed")
    return {"reddit_analysis": reply.content}

def synthesize_analyses(state: State):
    st.markdown('<div class="step-indicator">🔗 Synthesizing final answer...</div>', unsafe_allow_html=True)
    user_question = state.get("user_question") or ""
    google_analysis = state.get("google_analysis") or ""
    bing_analysis = state.get("bing_analysis") or ""
    reddit_analysis = state.get("reddit_analysis") or ""
    messages = get_synthesis_messages(user_question, google_analysis, bing_analysis, reddit_analysis)
    reply = safe_invoke(llm, messages)
    final_answer = reply.content
    st.success("✅ Research synthesis completed!")
    return {"final_answer": final_answer, "messages": [{"role": "assistant", "content": final_answer}]}

# ---------------- GRAPH ----------------
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

# ---------------- STREAMLIT ----------------
# --- app.py (UI section only) ---
import streamlit as st

def main():
    st.set_page_config(
        page_title="Multi-Source Research Agent",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write(
            "Searches Google, Bing, and Reddit, then synthesizes a single answer."
        )
        st.divider()
        st.subheader("Features")
        st.write(
            "- Parallel multi-source search\n"
            "- Intelligent filtering\n"
            "- Comprehensive synthesis\n"
            "- Real-time status"
        )

    # Header
    st.title("🔎 Multi-Source Research Agent")
    st.caption("Ask a question, watch the steps, and get a consolidated answer.")

    # Chat history
    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input (chat style)
    prompt = st.chat_input("What would you like to research?")
    if prompt:
        # Echo user message
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run pipeline with animated status
        with st.status("Running research…", expanded=True) as status:
            status.update(label="Searching sources (Google, Bing, Reddit)…", state="running")
            # Prepare initial state for the graph
            state = {
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
            }

            # Execute the research graph
            # (reuses your existing nodes and graph definition)
            status.write("• Running multi-source search")
            final_state = graph.invoke(state)

            status.update(label="Analyzing retrieved content…", state="running")
            status.write("• Parsing and ranking results")

            status.update(label="Synthesizing a final answer…", state="running")
            status.write("• Consolidating sources")

            status.update(label="Done", state="complete")

        # Render assistant response
        with st.chat_message("assistant"):
            if final_state.get("final_answer"):
                st.markdown(final_state["final_answer"])
            else:
                st.info("No answer was produced. Please try again with a different question.")

            # Details
            st.divider()
            st.subheader("Detailed Analysis")
            tab1, tab2, tab3 = st.tabs(["🌐 Google", "🔍 Bing", "📱 Reddit"])
            with tab1:
                st.markdown(final_state.get("google_analysis", "_No Google analysis available._"))
            with tab2:
                st.markdown(final_state.get("bing_analysis", "_No Bing analysis available._"))
            with tab3:
                st.markdown(final_state.get("reddit_analysis", "_No Reddit analysis available._"))

            # Optional: show selected Reddit sources
            selected_urls = final_state.get("selected_reddit_urls") or []
            if selected_urls:
                with st.expander("Reddit Sources"):
                    for i, url in enumerate(selected_urls, 1):
                        st.write(f"{i}. [{url}]({url})")

        # Persist assistant message in chat history
        st.session_state.chat.append({
            "role": "assistant",
            "content": final_state.get("final_answer", "_No answer produced._"),
        })

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
