# -*- coding: utf-8 -*-
"""Chatbot Groq_Rizky Octa Vianto.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NNQ3yYRxn2f3MvtDfyCPC4HPE8cZXOnu
"""

!pip install groq

import os
from groq import Groq

class Chatbot:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")  # Ambil dari environment variable
        if not self.api_key:
            raise ValueError("API Key is missing. Set GROQ_API_KEY as an environment variable.")
        self.client = Groq(api_key=self.api_key)

    def get_response(self, user_input):
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": user_input}],
            model="deepseek-r1-distill-llama-70b",
        )
        response = chat_completion.choices[0].message.content
        return response

if __name__ == "__main__":
    os.environ["GROQ_API_KEY"] = "YOUR API KEY HERE"  # Set API Key di runtime (sementara)
    chatbot = Chatbot()

    print("Chatbot Groq siap! Ketik 'exit' untuk keluar.")
    while True:
        user_input = input("Anda: ")
        if user_input.lower() == "exit":
            print("Chatbot: Sampai jumpa!")
            break
        response = chatbot.get_response(user_input)
        print(f"Chatbot: {response}")
