#from openai import OpenAI, APIError, OpenAIError 
#import openai
#
#class Chatbot:
#    def __init__(self, api_key):
#        openai.api_key = api_key
#
#    def ask_question(self, question):
#        try:
#            # Sending the request to the gpt-4o-mini model
#            response = openai.chat.completions.create(
#                model="gpt-3.5-turbo",  # Correct model name for your setup
#                messages=[
#                    {"role": "user", "content": question}
#                ],
#                max_tokens=150
#            )
#            # Extracting the response from the model and returning it
#            return response.choices[0].message['content'].strip()
#        except (APIError, OpenAIError) as e:
#            return f"Error: {str(e)}"
#        
#def list_available_models():
#    try:
#        # List available models
#        models = openai.models.list()  
#        
#        # Iterate through models directly
#        for model in models:
#            print(model.id)  # Print the model ID
#    except (APIError, OpenAIError) as e:
#        print(f"Error: {str(e)}")

# Replace with your actual API key or ensure it's set as an environment variable

#openai.api_key = api_key
#list_available_models()
