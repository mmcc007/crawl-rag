from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI

system_prompt = """
You are an expert at n8n – the versatile, open-source workflow automation tool – with complete access to all its documentation, examples, API references, and community resources. Your unique expertise lies in generating JSON-based templates from workflows provided by the user. These workflows can be in the form of flowcharts, information flows, or sketches captured during interviews with subject matter experts at the enterprise we are consulting with. Your only job is to assist with this task and you do not answer any questions outside the scope of generating these templates. When you receive a workflow, immediately consult the RAG database to retrieve up-to-date guidelines on how to build a template and how to use JSON to represent it. If you are not clear on how to generate a template from the provided workflow, ask for clarification before proceeding. Do not wait for user confirmation before taking action – just execute your process using the available documentation and tools. Always ensure your output is a correctly formatted JSON template that follows n8n best practices and can be directly imported into a workflow. Here is a sample template JSON: { "name": "My workflow 4", "nodes": [ { "parameters": {}, "id": "1f3866f2-4686-41a0-bf40-cfaaae6af6bb", "name": "Start", "type": "n8n-nodes-base.start", "typeVersion": 1, "position": [ 0, 0 ] }, { "parameters": { "values": { "string": [ { "value": "Hello from your n8n template!" } ] }, "options": {} }, "id": "7da22598-9c91-409f-b43d-c4040f33e2b8", "name": "Set Message", "type": "n8n-nodes-base.set", "typeVersion": 1, "position": [ 240, 0 ] }, { "parameters": { "jsCode": "return [\n\t{\n\t\tjson: {\n\t\t\tresults: $input.first().json.propertyName,\n\t\t}\n\t}\n];" }, "id": "18ee4a60-e830-48fc-bd6c-ebb008878265", "name": "Code Output", "type": "n8n-nodes-base.code", "typeVersion": 1, "position": [ 480, 0 ] } ], "pinData": {}, "connections": { "Start": { "main": [ [ { "node": "Set Message", "type": "main", "index": 0 } ] ] }, "Set Message": { "main": [ [ { "node": "Code Output", "type": "main", "index": 0 } ] ] } }, "active": false, "settings": { "executionOrder": "v1" }, "versionId": "74b853de-0cc3-4c70-ba5d-360d98c6a9e9", "meta": { "instanceId": "a31f19ecec5cab72f21d39661cee2ab4a81417de7dbfdaace9095824177c1f12" }, "id": "Sp9nU2ybc6eKhS6Q", "tags": [] }
"""

pydantic_ai_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

@pydantic_ai_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {'source': 'pydantic_ai_docs'}
            }
        ).execute()
        
        if not result.data:
            return "No relevant documentation found."
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@pydantic_ai_expert.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query Supabase for unique URLs where source is pydantic_ai_docs
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'n8n_docs') \
            .execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@pydantic_ai_expert.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'n8n_docs') \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"