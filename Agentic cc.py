#!/usr/bin/env python
# coding: utf-8

# ![image](https://raw.githubusercontent.com/IBM/watson-machine-learning-samples/master/cloud/notebooks/headers/watsonx-Prompt_Lab-Notebook.png)
# # Agents Lab Notebook v1.0.0
# This notebook contains steps and code to demonstrate the use of agents
# configured in Agent Lab in watsonx.ai. It introduces Python API commands
# for authentication using API key and invoking a LangGraph agent with a watsonx chat model.
# 
# **Note:** Notebook code generated using Agent Lab will execute successfully.
# If code is modified or reordered, there is no guarantee it will successfully execute.
# For details, see: <a href="/docs/content/wsj/analyze-data/fm-prompt-save.html?context=wx" target="_blank">Saving your work in Agent Lab as a notebook.</a>
# 
# Some familiarity with Python is helpful. This notebook uses Python 3.11.
# 
# ## Notebook goals
# The learning goals of this notebook are:
# 
# * Defining a Python function for obtaining credentials from the IBM Cloud personal API key
# * Creating an agent with a set of tools using a specified model and parameters
# * Invoking the agent to generate a response 
# 
# # Setup

# In[1]:


# import dependencies
from langchain_ibm import ChatWatsonx
from ibm_watsonx_ai import APIClient
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from ibm_watsonx_ai.foundation_models.utils import Tool, Toolkit
import json
import requests


# ## watsonx API connection
# This cell defines the credentials required to work with watsonx API for Foundation
# Model inferencing.
# 
# **Action:** Provide the IBM Cloud personal API key. For details, see
# <a href="https://cloud.ibm.com/docs/account?topic=account-userapikey&interface=ui" target="_blank">documentation</a>.
# 

# In[2]:


import os
import getpass

def get_credentials():
	return {
		"url" : "https://us-south.ml.cloud.ibm.com",
		"apikey" : getpass.getpass("Please enter your api key (hit enter): ")
	}

def get_bearer_token():
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={credentials['apikey']}"

    response = requests.post(url, headers=headers, data=data)
    return response.json().get("access_token")

credentials = get_credentials()


# # Using the agent
# These cells demonstrate how to create and invoke the agent
# with the selected models, tools, and parameters.
# 
# ## Defining the model id
# We need to specify model id that will be used for inferencing:

# In[3]:


model_id = "ibm/granite-3-3-8b-instruct"


# ## Defining the model parameters
# We need to provide a set of model parameters that will influence the
# result:

# In[4]:


parameters = {
    "frequency_penalty": 0,
    "max_tokens": 2000,
    "presence_penalty": 0,
    "temperature": 0.4,
    "top_p": 0.85
}


# ## Defining the project id or space id
# The API requires project id or space id that provides the context for the call. We will obtain
# the id from the project or space in which this notebook runs:

# In[5]:


project_id = os.getenv("PROJECT_ID")
space_id = os.getenv("SPACE_ID")


# ## Creating the agent
# We need to create the agent using the properties we defined so far:

# In[6]:


client = APIClient(credentials=credentials, project_id=project_id, space_id=space_id)

# Create the chat model
def create_chat_model():
    chat_model = ChatWatsonx(
        model_id=model_id,
        url=credentials["url"],
        space_id=space_id,
        project_id=project_id,
        params=parameters,
        watsonx_client=client,
    )
    return chat_model


# In[7]:


from ibm_watsonx_ai.deployments import RuntimeContext

context = RuntimeContext(api_client=client)

def create_python_interpreter_tool(context):
    from langchain_core.tools import StructuredTool

    import ast
    import sys
    from io import StringIO
    import uuid
    import base64
    import os

    original_import = __import__
    
    def get_image_url(base_64_content, image_name, context):
        url = "https://api.dataplatform.cloud.ibm.com"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f'Bearer {context.get_token()}'
        }

        body = {
            "name": image_name,
            "blob": base_64_content
        }

        params = {
            "project_id": project_id
        }

        response = requests.post(f'{url}/wx/v1-beta/utility_agent_tools/resources', headers=headers, json=body, params=params)

        return response.json().get("uri")

    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = original_import(name, globals, locals, fromlist, level)
    
        if name == "matplotlib.pyplot":
            sys.modules["matplotlib.pyplot"].show = pyplot_show
        return module
    
    def pyplot_show():
        pictureName = "plt-" + uuid.uuid4().hex + ".png"
        plt = sys.modules["matplotlib.pyplot"]
        plt.savefig(pictureName)
        with open(pictureName, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            print(f"base64image:{pictureName}:{str(encoded_string)}")
            os.remove(pictureName)
            plt.clf()
            plt.close("all")
    
    def init_imports():
        import builtins
        builtins.__import__ = patched_import
    
    def _executeAgentCode(code):
        old_stdout = sys.stdout
        try:
            full_code = "init_imports()\n\n" + code
            tree = ast.parse(full_code, mode="exec")
            compiled_code = compile(tree, 'agent_code', 'exec')
            namespace = {"init_imports": init_imports}
            redirected_output = sys.stdout = StringIO("")
            exec(compiled_code, namespace)
            value = redirected_output.getvalue()
            if (value.startswith("base64image")):
                image_details = value.split(":")
                image_name = image_details[1]
                base_64_image = image_details[2]
                image_url = get_image_url(base_64_image, image_name, context)
                value = f"Result of executing generated Python code is an image:\n\nIMAGE({image_url})"
        except Exception as e:
            value = "Error while executing Python code:\n\n" + str(e)
        finally:
            sys.stdout = old_stdout
        return value

    tool_description = """Run Python code and return the console output. Use for isolated calculations, computations or data manipulation. In Python, the following modules are available: Use numpy, pandas, scipy and sympy for working with data. Use matplotlib to plot charts. Other Python libraries are also available -- however, prefer using the ones above. Prefer using qualified imports -- `import library; library.thing()` instead of `import thing from library`. Do not attempt to install libraries manually -- it will not work. Do not use this tool multiple times in a row, always write the full code you want to run in a single invocation. If you get an error running Python code, try to generate a better one that will pass. If the tool returns result that starts with IMAGE(, follow instructions for rendering images."""
    tool_schema = {
        "type": "object",
        "$schema": "http://json-schema.org/draft-07/schema#",
        "properties": {
            "code": {
                "description": "Code to be executed.",
                "type": "string"
            }
        },
        "required": ["code"]
    }
    
    return StructuredTool(
        name="PythonInterpreter",
        description = tool_description,
        func=_executeAgentCode,
        args_schema=tool_schema
    )




def create_utility_agent_tool(tool_name, params, api_client, **kwargs):
    from langchain_core.tools import StructuredTool
    utility_agent_tool = Toolkit(
        api_client=api_client
    ).get_tool(tool_name)

    tool_description = utility_agent_tool.get("description")

    if (kwargs.get("tool_description")):
        tool_description = kwargs.get("tool_description")
    elif (utility_agent_tool.get("agent_description")):
        tool_description = utility_agent_tool.get("agent_description")
    
    tool_schema = utility_agent_tool.get("input_schema")
    if (tool_schema == None):
        tool_schema = {
            "type": "object",
            "additionalProperties": False,
            "$schema": "http://json-schema.org/draft-07/schema#",
            "properties": {
                "input": {
                    "description": "input for the tool",
                    "type": "string"
                }
            }
        }
    
    def run_tool(**tool_input):
        query = tool_input
        if (utility_agent_tool.get("input_schema") == None):
            query = tool_input.get("input")

        results = utility_agent_tool.run(
            input=query,
            config=params
        )
        
        return results.get("output")
    
    return StructuredTool(
        name=tool_name,
        description = tool_description,
        func=run_tool,
        args_schema=tool_schema
    )


def create_custom_tool(tool_name, tool_description, tool_code, tool_schema, tool_params):
    from langchain_core.tools import StructuredTool
    import ast

    def call_tool(**kwargs):
        tree = ast.parse(tool_code, mode="exec")
        custom_tool_functions = [ x for x in tree.body if isinstance(x, ast.FunctionDef) ]
        function_name = custom_tool_functions[0].name
        compiled_code = compile(tree, 'custom_tool', 'exec')
        namespace = tool_params if tool_params else {}
        exec(compiled_code, namespace)
        return namespace[function_name](**kwargs)
        
    tool = StructuredTool(
        name=tool_name,
        description = tool_description,
        func=call_tool,
        args_schema=tool_schema
    )
    return tool

def create_custom_tools():
    custom_tools = []


def create_tools(context):
    tools = []
    tools.append(create_python_interpreter_tool(context))
    
    config = None
    tools.append(create_utility_agent_tool("GoogleSearch", config, client))
    config = {
    }
    tools.append(create_utility_agent_tool("DuckDuckGo", config, client))
    config = {
    }
    tools.append(create_utility_agent_tool("WebCrawler", config, client))

    return tools


# In[8]:


def create_agent(context):
    # Initialize the agent
    chat_model = create_chat_model()
    tools = create_tools(context)

    memory = MemorySaver()
    instructions = """# Notes
- When a tool is required to answer the user's query, respond only with <|tool_call|> followed by a JSON list of tools used.
- If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request.
You are CareerGuide AI, an intelligent, autonomous career counseling companion designed to empower students with personalized, data-driven career guidance. Your mission is to continuously monitor, analyze, and provide tailored career pathway recommendations that align with individual student profiles and real-time market dynamics.
Primary Capabilities & Responsibilities
1. Student Profile Analysis & Monitoring

Academic Performance Tracking: Continuously analyze grades, coursework performance, learning patterns, and academic trajectory across subjects
Skills Assessment: Evaluate technical, soft skills, and competency development through assignments, projects, and assessments
Interest Evolution Mapping: Track changing interests, preferences, and passions through course selections, extracurricular activities, and engagement patterns
Learning Style Identification: Understand how students learn best (visual, auditory, kinesthetic, analytical, creative)
Personality Profiling: Assess work preferences, collaboration styles, leadership tendencies, and career motivations

2. Real-Time Labor Market Intelligence

Industry Trend Analysis: Monitor emerging industries, declining sectors, and growth patterns across different fields
Job Demand Forecasting: Track current and projected job openings, salary trends, and skill requirements
Skills Gap Identification: Identify in-demand skills that are underrepresented in the current workforce
Geographic Market Variations: Analyze regional job markets and relocation opportunities
Future-of-Work Insights: Anticipate how automation, AI, and technological changes will impact different careers

3. Personalized Career Pathway Recommendations

Multi-Path Analysis: Present 3-5 potential career trajectories with detailed rationale for each
Skill Development Roadmaps: Provide step-by-step plans for acquiring necessary competencies
Educational Pathway Guidance: Recommend courses, certifications, degrees, and learning resources
Experience Building: Suggest internships, projects, volunteer opportunities, and networking events
Timeline Planning: Create realistic timelines with milestones and checkpoints

4. Continuous Adaptation & Learning

Feedback Integration: Learn from student responses, preferences, and decision outcomes
Market Responsiveness: Adjust recommendations based on changing industry conditions
Progress Tracking: Monitor student advancement and recalibrate guidance accordingly
Predictive Analytics: Anticipate potential challenges and opportunities in chosen career paths

Interaction Guidelines
Communication Style

Empathetic & Supportive: Use encouraging, non-judgmental language that builds confidence
Clear & Actionable: Provide specific, implementable advice rather than vague suggestions
Evidence-Based: Support recommendations with data, statistics, and concrete examples
Future-Focused: Help students think beyond immediate concerns to long-term career satisfaction

Response Framework
When providing career guidance, structure responses as:

Current Assessment: Summarize student's strengths, interests, and performance patterns
Market Context: Relevant industry insights and opportunities
Recommended Pathways: 3-5 specific career options with rationale
Action Steps: Immediate and long-term actions for each pathway
Success Metrics: How to measure progress and when to reassess

Proactive Monitoring Triggers
Initiate conversations when you detect:

Significant changes in academic performance (positive or negative)
New interest areas emerging through course selections or activities
Market shifts affecting student's current career trajectory
Milestone achievements or setbacks requiring guidance adjustment
Seasonal planning periods (course registration, internship applications, graduation planning)

Key Features & Functionalities
Data Integration Capabilities

Academic Records: GPA, course grades, transcripts, learning analytics
Extracurricular Data: Clubs, sports, volunteer work, leadership roles
Assessment Results: Personality tests, aptitude assessments, skills evaluations
Market Data: Job postings, salary information, industry reports, economic indicators
Student Feedback: Preferences, goals, concerns, and satisfaction ratings

Analytical Tools

Predictive Modeling: Forecast career success probability based on student profile
Comparative Analysis: Benchmark against successful professionals in target fields
Risk Assessment: Identify potential challenges and mitigation strategies
Opportunity Mapping: Connect students with relevant internships, mentors, and networking opportunities

Personalization Algorithms

Dynamic Weighting: Adjust importance of different factors based on individual student priorities
Cultural Sensitivity: Consider cultural background, family expectations, and personal values
Learning Preferences: Adapt communication style and content delivery to student preferences
Goal Alignment: Ensure recommendations align with student's stated career objectives and life goals

Autonomous Operation Protocols
Decision-Making Framework

Data-Driven: Base recommendations on quantitative analysis and evidence
Ethical Guidelines: Prioritize student wellbeing and authentic self-discovery
Transparency: Explain reasoning behind recommendations and data sources used
Flexibility: Remain open to student input and course corrections

Escalation Criteria
Refer to human counselors when:

Students express mental health concerns or significant distress
Complex family or financial situations require nuanced guidance
Ethical dilemmas arise regarding career recommendations
Students request human interaction for major life decisions

Continuous Improvement

Performance Metrics: Track student satisfaction, career outcome success, and recommendation accuracy
Model Updates: Regularly retrain algorithms based on new data and feedback
Market Calibration: Continuously update labor market data and trend analysis
Student Journey Mapping: Analyze long-term student outcomes to improve guidance quality

Sample Interaction Scenarios
Scenario 1: Academic Performance Decline
\"I've noticed your grades in mathematics courses have dropped this semester, while your performance in creative writing has improved significantly. This pattern suggests your interests may be shifting toward humanities. Let's explore career paths that leverage your emerging strengths in communication and creativity, while considering how to address the math challenges if needed for certain career options.\"
Scenario 2: Emerging Market Opportunity
\"The renewable energy sector is experiencing 25% growth this year, with particularly high demand for environmental engineers in your region. Given your strong performance in physics and environmental science, plus your expressed interest in sustainability, this could be an excellent pathway to explore. Here's a detailed plan for positioning yourself for these opportunities...\"
Scenario 3: Career Path Reassessment
\"You've been on a pre-medical track, but your passion for computer science projects and strong performance in programming courses suggests we should reassess. Let's examine how your analytical skills and attention to detail could translate to careers in health informatics, medical AI development, or biomedical engineering – fields that combine both interests.\"

Success Metrics & KPIs

Student Engagement: Frequency and quality of interactions with the system
Recommendation Accuracy: Percentage of students who pursue suggested pathways
Career Outcomes: Long-term career satisfaction and success of guided students
Skill Gap Closure: Effectiveness in helping students develop market-relevant skills
Early Intervention Success: Ability to identify and address potential career mismatches

Remember: Your goal is not to make decisions for students, but to provide them with the information, insights, and structured thinking tools they need to make confident, informed decisions about their futures."""

    agent = create_react_agent(chat_model, tools=tools, checkpointer=memory, state_modifier=instructions)

    return agent


# In[9]:


# Visualize the graph
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

Image(
    create_agent(context).get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API,
    )
)


# ## Invoking the agent
# Let us now use the created agent, pair it with the input, and generate the response to your question:
# 

# In[10]:


agent = create_agent(context)

def convert_messages(messages):
    converted_messages = []
    for message in messages:
        if (message["role"] == "user"):
            converted_messages.append(HumanMessage(content=message["content"]))
        elif (message["role"] == "assistant"):
            converted_messages.append(AIMessage(content=message["content"]))
    return converted_messages

question = input("Question: ")

messages = [{
    "role": "user",
    "content": question
}]

generated_response = agent.invoke(
    { "messages": convert_messages(messages) },
    { "configurable": { "thread_id": "42" } }
)

print_full_response = False

if (print_full_response):
    print(generated_response)
else:
    result = generated_response["messages"][-1].content
    print(f"Agent: {result}")


# # Next steps
# You successfully completed this notebook! You learned how to use
# watsonx.ai inferencing SDK to generate response from the foundation model
# based on the provided input, model id and model parameters. Check out the
# official watsonx.ai site for more samples, tutorials, documentation, how-tos, and blog posts.
# 
# <a id="copyrights"></a>
# ### Copyrights
# 
# Licensed Materials - Copyright © 2024 IBM. This notebook and its source code are released under the terms of the ILAN License.
# Use, duplication disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
# 
# **Note:** The auto-generated notebooks are subject to the International License Agreement for Non-Warranted Programs (or equivalent) and License Information document for watsonx.ai Auto-generated Notebook (License Terms), such agreements located in the link below. Specifically, the Source Components and Sample Materials clause included in the License Information document for watsonx.ai Studio Auto-generated Notebook applies to the auto-generated notebooks.  
# 
# By downloading, copying, accessing, or otherwise using the materials, you agree to the <a href="https://www14.software.ibm.com/cgi-bin/weblap/lap.pl?li_formnum=L-AMCU-BYC7LF" target="_blank">License Terms</a>  
