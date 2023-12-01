import g4f

g4f.debug.logging = True  # Enable logging
g4f.check_version = False  # Disable automatic version checking
print(g4f.version)  # Check version
print(g4f.Provider.Ails.params)  # Supported args

# Automatic selection of provider

# Streamed completion
response = g4f.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "what are the benefits of papaya"}],
    provider=g4f.Provider.Phind,
    # stream=True,
)

# for message in response:
#     print(message, flush=True, end='')

# # Normal response
# response = g4f.ChatCompletion.create(
#     model=g4f.models.gpt_4,
#     messages=[{"role": "user", "content": "what are the benifits of neem"}],
# )  # Alternative model setting

print(response)