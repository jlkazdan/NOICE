from openai import OpenAI

client = OpenAI()

print(client.fine_tuning.jobs.create(
    training_file="file-CA9AZ7i2uJP6sftN8mswQt", model="gpt-4o-mini-2024-07-18"
))
