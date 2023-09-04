# Generative Agents with Guidance, Langchain, and local LLMs
This is the implementation of paper ["Generative Agents: Interactive Simulacra of Human Behavior"](https://arxiv.org/pdf/2304.03442.pdf). This is based on the [Langchain implementation](https://python.langchain.com/en/latest/use_cases/agent_simulations/characters.html). We improve and add more features to make it like the original paper as much as possible.
For more explaination, please check [my medium post](https://medium.com/@gartist/implement-generative-agent-with-local-llm-guidance-and-langchain-full-features-fa57655f3de1).

Note that: I just fixed a conflict between current prompts and the guidance. I recommend using the guidance==0.063 because of the stability.

### Supported Features:
- [x] Work with local LLM
- [x] Memory and Retrieval
- [x] Reflection
- [x] Planning (need to improve)
- [x] Reacting and re-planning
- [x] Dialogue generation (need to improve)
- [x] Agent summary
- [x] Interview
- [ ] Web UI (Gradio)

# How to use
### Install
Python packages:
- [Guidance](https://github.com/microsoft/guidance) `pip install guidance==0.0.63`
- [GPTQ-for-LLaMa](https://github.com/oobabooga/GPTQ-for-LLaMa.git)
- [Langchain](https://github.com/hwchase17/langchain) `pip install langchain==0.0.190`
- [Faiss](https://github.com/facebookresearch/faiss) (For VectorStore, feel free to change to your own VectorStore at [this link](https://python.langchain.com/en/latest/modules/indexes/vectorstores.html))

The GPTQ-for-LLaMa I used is the oobabooga's fork. You can install it with [this command](https://github.com/oobabooga/text-generation-webui/blob/main/docs/GPTQ-models-(4-bit-mode).md#step-1-install-gptq-for-llama).

### Run
Please check the notebook file. I use the [wizard-mega-13B-GPTQ](https://huggingface.co/TheBloke/wizard-mega-13B-GPTQ) model. Feel free to try others.

# Notebook
### Define Generative Agent
```python
description = "Sam is a Ph.D student, his major is CS;Sam likes computer;Sam lives with his friend, Bob;Sam's farther is a doctor;Sam has a dog, named Max"
sam = GenerativeAgent(guidance=guidance, 
                      name='Sam',
                      age=23, 
                      des=description, 
                      trails='funny, like football, play CSGO', 
                      embeddings_model=embeddings_model)
```
### Add memories
```python
sam_observations = [
    "Sam wake up in the morning",
    "Sam feels tired because of playing games",
    "Sam has a assignment of AI course",
    "Sam see Max is sick",
    "Bob say hello to Sam",
    "Bob leave the room",
    "Sam say goodbye to Bob",
]
sam.add_memories(sam_observations)
```
### Summary
```python
summary = sam.get_summary(force_refresh=True)
print(summary)
"""
Name: Sam (age: 23)
Summary: Sam can be described as a Ph.D student who is interested in computer science and has a dog named Max. He is also a student of AI course and has a father who is a doctor. Sam is also a gamer and lives with his friend Bob. Additionally, Sam is a caring person who feels tired due to playing games and says goodbye to his friend Bob.. Sam is a Ph.D student majoring in Computer Science. He wakes up in the morning and lives with his friend Bob. Sam has a dog named Max and he is currently feeling tired due to playing games. Sam also has an assignment for his AI course.. it is difficult to determine Sam's feeling about his recent progress in life. However, if we assume that Sam is satisfied with his progress, we can describe his feeling as content or fulfilled.
"""
```
### Planning and update status
```python
status = sam.update_status()
```
![alt text](https://github.com/QuangBK/generativeAgent_LLM/blob/main/imgs/planning.png?raw=true)

### Reaction
```python
bool_react, reaction, context = sam.react(observation='The dog bowl is empty', 
                                          observed_entity='Dog bowl', 
                                          entity_status='The dog bowl is empty')
print(f"{bool_react}\nReaction: {reaction}\nContext: {context}")
"""
Yes
Reaction: Sam could put food in the dog's bowl and then call Max over to eat.
Context: Sam has a dog named Max, and he is a Ph.D student majoring in CS. Sam's father is a doctor, and Sam lives with his friend Bob. Sam likes computers and is currently taking an AI course. Sam is tired because of playing games. Bob left the room and said hello to Sam. Sam woke up in the morning and saw that the dog bowl was empty.
"""    
```
![alt text](https://github.com/QuangBK/generativeAgent_LLM/blob/main/imgs/reaction.png?raw=true)

### Dialogue generation
```python
bool_react, reaction, context = sam.react(observation='Bob come room with a new PC', 
                                          observed_entity=bob,
                                          entity_status='Bob is setting up his new PC')

print(sam.dialogue_list[0])
"""
Friday June 02, 2023, 18:15

Bob: Hey Sam, check this out! I got a new PC and it's amazing.
Sam: That's great, Bob. Do you need any help setting it up?
Bob: No, I got it all set up already. It's just for gaming, but I'm really excited.
Sam: That's awesome. I'm always interested in trying out new hardware. Maybe I'll stop by and check it out.
Bob: Yeah, of course. I was just thinking, maybe you could help me with a few settings. I'm not that great at this stuff.
Sam: Sure, I'd be happy to help. When do you want to get started?
Bob: How about later on tonight? I'll call you when I'm ready to get going.
Sam: No problem. Let me know when you're ready and I'll head over.
"""
```

### Interview
```python
response = sam.interview('Friend', 'Who do you live with?')
print(response)
"""
I live with a friend of mine, his name is Bob.
"""
```

# License

Creative Commons Attribution-NonCommercial (CC BY-NC-4.0) 
