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
    st.write(f"🔍 Searching Google for: {user_question}")
    google_results = serp_search(user_question, engine="google")
    return {"google_results": google_results}

def bing_search(state: State):
    user_question = state.get("user_question", "")
    st.write(f"🔍 Searching Bing for: {user_question}")
    bing_results = serp_search(user_question, engine="bing")
    return {"bing_results": bing_results}

def reddit_search(state: State):
    user_question = state.get("user_question", "")
    st.write(f"🔍 Searching Reddit for: {user_question}")
    reddit_results = reddit_search_api(keyword=user_question)
    return {"reddit_results": reddit_results}

def analyze_reddit_posts(state: State):
    user_question = state.get("user_question", "")
    reddit_results = state.get("reddit_results", "")

    if not reddit_results:
        return {"selected_reddit_urls": []}

    structured_llm = llm.with_structured_output(RedditURLAnalysis)
    messages = get_reddit_url_analysis_messages(user_question, reddit_results)

    try:
        analysis = safe_invoke(structured_llm, messages)
        selected_urls = analysis.selected_urls
        st.write("✅ Selected URLs:")
        for i, url in enumerate(selected_urls, 1):
            st.write(f"{i}. {url}")
    except Exception as e:
        st.write(f"⚠️ Error selecting URLs: {e}")
        selected_urls = []

    return {"selected_reddit_urls": selected_urls}

def retrieve_reddit_posts(state: State):
    st.write("📥 Getting reddit post comments")
    selected_urls = state.get("selected_reddit_urls", [])
    if not selected_urls:
        return {"reddit_post_data": []}

    reddit_post_data = reddit_post_retrieval(selected_urls)
    if reddit_post_data:
        st.write(f"✅ Got {len(reddit_post_data)} posts")
    else:
        st.write("⚠️ Failed to get post data")
        reddit_post_data = []
    return {"reddit_post_data": reddit_post_data}

def analyze_google_results(state: State):
    st.write("🧠 Analyzing Google search results")
    user_question = state.get("user_question", "")
    google_results = state.get("google_results", "")
    messages = get_google_analysis_messages(user_question, google_results)
    reply = safe_invoke(llm, messages)
    return {"google_analysis": reply.content}

def analyze_bing_results(state: State):
    st.write("🧠 Analyzing Bing search results")
    user_question = state.get("user_question", "")
    bing_results = state.get("bing_results", "")
    messages = get_bing_analysis_messages(user_question, bing_results)
    reply = safe_invoke(llm, messages)
    return {"bing_analysis": reply.content}

def analyze_reddit_results(state: State):
    st.write("🧠 Analyzing Reddit search results")
    user_question = state.get("user_question", "")
    reddit_results = state.get("reddit_results", "")
    reddit_post_data = state.get("reddit_post_data", "")
    messages = get_reddit_analysis_messages(user_question, reddit_results, reddit_post_data)
    reply = safe_invoke(llm, messages)
    return {"reddit_analysis": reply.content}

def synthesize_analyses(state: State):
    st.write("🔗 Combining all analyses...")
    user_question = state.get("user_question", "")
    google_analysis = state.get("google_analysis", "")
    bing_analysis = state.get("bing_analysis", "")
    reddit_analysis = state.get("reddit_analysis", "")
    messages = get_synthesis_messages(user_question, google_analysis, bing_analysis, reddit_analysis)
    reply = safe_invoke(llm, messages)
    final_answer = reply.content
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
def main():
    st.title("🔎 Multi-Source Research Agent")
    user_input = st.text_input("Ask me anything:")
    if st.button("Search"):
        if user_input:
            state = {
                "messages": [{"role": "user", "content": user_input}],
                "user_question": user_input,
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
            st.write("🚀 Starting research it will take 3 to 5 minutes....")
            with st.spinner('Working hard...'):
                final_state = graph.invoke(state)
            if final_state.get("final_answer"):
                st.subheader("✅ Final Answer:")
                st.write(final_state.get('final_answer'))
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
