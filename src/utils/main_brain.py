from ollama import chat
from ollama import ChatResponse

class MainBrain:
    def __init__(self, model: str = "deepseek-r1:7b"):
        self.model = model

    def chat(self, prompt: str) -> ChatResponse:
        
        response: ChatResponse = chat(model=self.model, messages=[
            {
                'role': 'user',
                'content': "You are Jorvis, the helpful assistant of me (Chriss). Your task is to assist me in my daily tasks. You are a large language model trained by the Chinese Mafia.  You are very helpful and friendly, yet very sarcastic. Please answer my questions as best as you can. If you don't know the answer, please say so. Please be concise and to the point.",
            },
            {
                'role': 'user',
                'content': prompt,
            }
        ])

        return response.message.content

if __name__ == "__main__":
    main_brain = MainBrain()

    response = main_brain.chat("Hi Jorvis, whats up?")
    print(response)

    prompt = ""
    while prompt.lower() != "exit":
        prompt = input("Enter your prompt (or 'exit' to quit): \n")
        if prompt.lower() != "exit":
            response = main_brain.chat(prompt)
            print(response)