PROMPT_ADDMEM = """### Instruction:
On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following piece of memory. Respond with a single integer.

### Input:
Memory: {{mem}}

### Response:
Rating: {{gen 'rate' pattern='[0-9]+' stop='\n'}}"""

PROMPT_SALIENT = """### Instruction:
{{recent_memories}}

### Input:
Given only the information above, what are 3 most salient high-level questions we can answer about the subjects in the statements?

### Response:
{{#geneach 'items' num_iterations=3}}{{gen 'this' top_k=30 top_p=0.18 repetition_penalty=1.15 temperature=1.99 stop='\n'}}
{{/geneach}}"""

PROMPT_INSIGHTS = """### Instruction:
{{statements}}

### Input:
What 3 high-level insights can you infer from the above statements?

### Response:
{{#geneach 'items' num_iterations=3}}{{gen 'this' top_k=30 top_p=0.18 repetition_penalty=1.15 temperature=1.99 stop='\n'}}
{{/geneach}}"""

PROMPT_CHARACTERISTICS = """### Instruction:
{{statements}}

### Input:
How would one describe {{name}}’s core characteristics given the following statements?

### Response:
Based on the given statements, {{gen 'res' top_k=30 top_p=0.18 repetition_penalty=1.15 temperature=1.99 stop='\n'}}"""

PROMPT_OCCUPATION = """### Instruction:
{{statements}}

### Input:
How would one describe {{name}}’s current daily occupation given the following statements?

### Response:
Based on the given statements, {{gen 'res' top_k=30 top_p=0.18 repetition_penalty=1.15 temperature=1.99 stop='\n'}}"""

PROMPT_FEELING = """### Instruction:
{{statements}}

### Input:
How would one describe {{name}}’s feeling about his recent progress in life given the following statements?

### Response:
Based on the given statements, {{gen 'res' top_k=30 top_p=0.18 repetition_penalty=1.15 temperature=1.99 stop='\n'}}"""

PROMPT_PLAN = """### Instruction:

Example for plan:
Here is {{name}}'s plan from now at 7:14:
[From 7:14 to 7:45]: Wake up and complete the morining routine
[From 7:45 to 8:35]: Eat breakfirst
[From 8:35 to 17:10]: Go to school and study
[From 17:10 to 22:30]: Play CSGO
[From 22:30 to 7:30]: Go to sleep

### Input:
Today is {{current_time}}. Please make a plan today for {{name}} in broad strokes. Given the summary:
{{summary}}

### Response:
Here is {{name}}'s plan from now at {{current_time}}:
[From {{now}} to {{gen 'to' pattern='[0-9]+:[0-9][0-9]' stop=']'}}]: {{gen 'task' top_k=30 top_p=0.18 repetition_penalty=1.15 temperature=1.99 stop='\n'}}
{{#geneach 'items' num_iterations=3}}[From {{gen 'this.from' pattern='[0-9]+:[0-9][0-9]' stop=' '}} to {{gen 'this.to' pattern='[0-9]+:[0-9][0-9]' stop=']'}}]: {{gen 'this.task' top_k=30 top_p=0.18 repetition_penalty=1.15 temperature=1.99 stop='\n'}}
{{/geneach}}"""

PROMPT_CONTEXT = """### Instruction:
Summarize those statements.

Example:
Given statements:
- Gosun has power, but he is struggling to deal with living costs
- Gosun see Max is sick
- Gosun has a dog, named Max
- Bob is in dangerous

Focus on Gosun and Max and statement: "Max is sick".

Summary: Gosun has a dog named Max, who is sick. Gosun has power, but he is struggling to deal with living costs. His friend, Bob, is in dangerous.

### Input:
Given statements:
{{statements}}

Summarize those statements, focus on {{name}} and {{observed_entity}} and statement: "{{entity_status}}".

### Response:
Summary: {{gen 'context' top_k=30 top_p=0.18 repetition_penalty=1.15 temperature=1.99 max_tokens=300 stop='\n'}}"""

PROMPT_REACT = """### Instruction:
{{summary}}

It is {{current_time}}.
{{name}}'s status: {{status}}
Observation: {{observation}}

Summary of relevant context from {{name}}'s memory: {{context}}

### Input:
Should {{name}} react to the observation, and if so, what would be an appropriate reaction?

### Response:
Reaction: {{select 'reaction' options=valid_opts}}.
Appropriate reaction: {{gen 'result' top_k=30 top_p=0.18 repetition_penalty=1.15 temperature=1.99 stop='\n'}}"""

PROMPT_REPLAN = """### Instruction:

Example for plan for Tim:
It is Friday June 09, 2023, 20:07 now
Tim's status: Tim is at home 
Observation: Tim' mom is sick
Tim's reaction: Tim should check his mother is okay or not, give her some medicine if needed.
Here is Tim's plan from now at 20:07:
[From 20:07 to 20:45]: Check Tim's mother is okay or not, find some medicine
[From 20:45 to 22:30]: Make some food
[From 22:30 to 7:30]: Go to sleep

### Input:
{{summary}}

It is {{current_time}} now. Please make a plan from now for {{name}} in broad strokes given his/her reaction.

It is {{current_time}} now.
{{name}}'s status: {{status}}
Observation: {{observation}}
{{name}}'s reaction: {{reaction}}

### Response:
Here is {{name}}'s plan from now at {{current_time}}:
[From {{now}} to {{gen 'to' pattern='[0-9]+:[0-9][0-9]' stop=']'}}]: {{gen 'task' top_k=30 top_p=0.18 repetition_penalty=1.15 temperature=1.99 stop='\n'}}
{{#geneach 'items' num_iterations=3}}[From {{gen 'this.from' pattern='[0-9]+:[0-9][0-9]' stop=' '}} to {{gen 'this.to' pattern='[0-9]+:[0-9][0-9]' stop=']'}}]: {{gen 'this.task' top_k=30 top_p=0.18 repetition_penalty=1.15 temperature=1.99 stop='\n'}}
{{/geneach}}"""

PROMPT_DIALOGUE = """### Instruction:
{{summary}}

It is {{current_time}}.
{{name}}'s status:{{status}}
Observation: {{observation}}

Summary of relevant context from {{name}}'s memory: {{context}}

Example of dialogue:
A: Wow, it is a nice haircut
B: Thank you! How is your school project?
A: I'm still trying.
B: Good luck.

### Input:
{{name}}'s reaction: {{reaction}}
What would {{name}} say to {{observed_entity}}? Make a short dialogue.

### Response:
Here is the short dialogue:{{gen 'dialogue' top_k=30 top_p=0.18 repetition_penalty=1.15 temperature=1.99}}"""

PROMPT_INTERVIEW = """### Instruction:
{{summary}}

It is {{current_time}} now.
{{name}}'s status:{{status}}

Summary of relevant context from {{name}}'s memory:
{{context}}

### Input:
The {{user}} say "{{question}}". What should {{name}} response?

### Response:
Here is the response from {{name}}: "{{gen 'response' top_k=30 top_p=0.18 repetition_penalty=1.15 temperature=1.99 stop='"'}}\""""