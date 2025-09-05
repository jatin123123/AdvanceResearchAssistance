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
def main():
    # Page config
    st.set_page_config(
        page_title="Multi-Source Research Agent",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .search-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .result-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-top: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .step-indicator {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🔎 Multi-Source Research Agent</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This research agent searches across multiple sources:
        - 🌐 **Google Search**: Web results and knowledge panels
        - 🔍 **Bing Search**: Additional web perspectives  
        - 📱 **Reddit**: Community discussions and insights
        
        The AI analyzes all sources and provides a comprehensive answer.
        """)
        
        st.header("⚡ Features")
        st.write("""
        - Parallel multi-source searching
        - Intelligent content filtering
        - Comprehensive analysis synthesis
        - Real-time progress tracking
        """)
        
        st.header("⏱️ Processing Time")
        st.info("Research typically takes 3-5 minutes")
    
    # Main content area
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="search-box">', unsafe_allow_html=True)
        st.subheader("🤔 What would you like to research?")
        user_input = st.text_input(
            "Enter your question:",
            placeholder="e.g., What are the latest developments in AI?",
            label_visibility="collapsed"
        )
        
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            search_button = st.button("🚀 Start Research", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Results area
    if search_button:
        if user_input:
            # Initialize progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create containers for live updates
            search_container = st.container()
            analysis_container = st.container()
            
            state: State = {
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
            
            # Step indicators
            with search_container:
                st.markdown("### � Search Phase")
                search_cols = st.columns(3)
                
                with search_cols[0]:
                    google_status = st.empty()
                with search_cols[1]:
                    bing_status = st.empty()
                with search_cols[2]:
                    reddit_status = st.empty()
            
            with analysis_container:
                st.markdown("### 🧠 Analysis Phase")
                analysis_cols = st.columns(3)
                
                with analysis_cols[0]:
                    google_analysis_status = st.empty()
                with analysis_cols[1]:
                    bing_analysis_status = st.empty()
                with analysis_cols[2]:
                    reddit_analysis_status = st.empty()
            
            # Execute research
            status_text.text("🚀 Initializing research...")
            progress_bar.progress(10)
            
            try:
                final_state = graph.invoke(state)
                progress_bar.progress(100)
                status_text.text("✅ Research completed!")
                
                # Display results
                if final_state.get("final_answer"):
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown("## 🎯 Research Results")
                    st.markdown("---")
                    st.markdown(final_state.get('final_answer'))
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Additional insights in expandable sections
                    with st.expander("📊 Detailed Analysis Breakdown"):
                        tab1, tab2, tab3 = st.tabs(["🌐 Google Analysis", "🔍 Bing Analysis", "📱 Reddit Analysis"])
                        
                        with tab1:
                            if final_state.get("google_analysis"):
                                st.write(final_state.get("google_analysis"))
                            else:
                                st.write("No Google analysis available")
                        
                        with tab2:
                            if final_state.get("bing_analysis"):
                                st.write(final_state.get("bing_analysis"))
                            else:
                                st.write("No Bing analysis available")
                        
                        with tab3:
                            if final_state.get("reddit_analysis"):
                                st.write(final_state.get("reddit_analysis"))
                            else:
                                st.write("No Reddit analysis available")
                    
                    # Show selected Reddit URLs if available
                    selected_urls = final_state.get("selected_reddit_urls")
                    if selected_urls:
                        with st.expander("🔗 Reddit Sources"):
                            for i, url in enumerate(selected_urls, 1):
                                st.write(f"{i}. [{url}]({url})")
                
            except Exception as e:
                st.error(f"❌ An error occurred during research: {str(e)}")
                progress_bar.progress(0)
                status_text.text("❌ Research failed")
                
        else:
            st.warning("⚠️ Please enter a question to start your research!")

if __name__ == "__main__":
    main()
