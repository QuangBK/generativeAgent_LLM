PROMPT_ADDMEM = """### Instruction:
On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following piece of memory. Respond with a single integer.

### Input:
Memory: {{mem}}

### Response:
Rating: {{gen 'rate' pattern='[0-9]+' stop='\\n'}}"""

PROMPT_SALIENT = """### Instruction:
{{recent_memories}}

### Input:
Given only the information above, what are 3 most salient high-level questions we can answer about the subjects in the statements?

### Response:
{{#geneach 'items' num_iterations=3}}{{gen 'this' stop='\\n'}}
{{/geneach}}"""

PROMPT_INSIGHTS = """### Instruction:
{{statements}}

### Input:
What 3 high-level insights can you infer from the above statements?

### Response:
{{#geneach 'items' num_iterations=3}}{{gen 'this' stop='\\n'}}
{{/geneach}}"""

PROMPT_CHARACTERISTICS = """### Instruction:
{{statements}}

### Input:
How would one describe {{name}}’s core characteristics given the following statements?

### Response:
Based on the given statements, {{gen 'res' stop='\\n'}}"""

PROMPT_OCCUPATION = """### Instruction:
{{statements}}

### Input:
How would one describe {{name}}’s current daily occupation given the following statements?

### Response:
Based on the given statements, {{gen 'res' stop='\\n'}}"""

PROMPT_FEELING = """### Instruction:
{{statements}}

### Input:
How would one describe {{name}}’s feeling about his recent progress in life given the following statements?

### Response:
Based on the given statements, {{gen 'res' stop='\\n'}}"""

PROMPT_PLAN = """### Instruction:
{{summary}}

Example for plan:
Here is {{name}}'s plan from now at 7:14:
[From 7:14 to 7:45]: Wake up and complete the morining routine
[From 7:45 to 8:35]: Eat breakfirst
[From 8:35 to 17:10]: Go to school and study
[From 17:10 to 22:30]: Play CSGO
[From 22:30 to 7:30]: Go to sleep

### Input:
Today is {{current_time}}. Please make a plan today for {{name}} in broad strokes.

### Response:
Here is {{name}}'s plan from now at {{current_time}}:
[From {{now}} to {{gen 'to' pattern='[0-9]+:[0-9][0-9]' stop=' ]'}}]: {{gen 'task' temperature=0.95 stop='\\n'}}
{{#geneach 'items' num_iterations=3}}[From {{gen 'this.from' pattern='[0-9]+:[0-9][0-9]' stop=' '}} to {{gen 'this.to' pattern='[0-9]+:[0-9][0-9]' stop=' ]'}}]: {{gen 'this.task' temperature=0.95 stop='\\n'}}
{{/geneach}}"""

PROMPT_CONTEXT = """### Instruction:
{{statements}}

### Input:
Summarize those statements, focus on {{name}} and {{observed_entity}} and statement: "{{entity_status}}".

### Response:
Based on the given statements, it can be summarized as follows:

{{gen 'context' max_tokens=300 stop='\\n'}}"""

PROMPT_REACT = """### Instruction:
{{summary}}

It is {{current_time}}.
{{name}}'s status:{{status}}
Observation: {{observation}}

Summary of relevant context from {{name}}'s memory: {{context}}

### Input:
Should {{name}} react to the observation, and if so, what would be an appropriate reaction?

### Response:
Reaction: {{#select 'reaction'}}Yes{{or}}No{{/select}}.
Appropriate reaction: {{gen 'result' temperature=0.95 stop='\\n'}}"""

PROMPT_REPLAN = """### Instruction:
{{summary}}

It is {{current_time}} now.
{{name}}'s status:{{status}}
Observation: {{observation}}
Reaction: {{reaction}}

Example for plan:
Here is {{name}}'s plan from now at 7:14:
[From 7:14 to 7:45]: Wake up and complete the morining routine
[From 7:45 to 8:35]: Eat breakfirst
[From 8:35 to 17:10]: Go to school and study
[From 17:10 to 22:30]: Play CSGO
[From 22:30 to 7:30]: Go to sleep

### Input:
It is {{current_time}} now. Please make a plan from now for {{name}} in broad strokes given his reaction.

### Response:
Here is {{name}}'s plan from now at {{current_time}}:
[From {{now}} to {{gen 'to' pattern='[0-9]+:[0-9][0-9]' stop=' ]'}}]: {{gen 'task' temperature=0.95 stop='\\n'}}
{{#geneach 'items' num_iterations=3}}[From {{gen 'this.from' pattern='[0-9]+:[0-9][0-9]' stop=' '}} to {{gen 'this.to' pattern='[0-9]+:[0-9][0-9]' stop=' ]'}}]: {{gen 'this.task' temperature=0.95 stop='\\n'}}
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
Here is the short dialogue:{{gen 'dialogue' temperature=0.95 stop=''}}"""

PROMPT_INTERVIEW = """### Instruction:
{{summary}}

It is {{current_time}}.
{{name}}'s status:{{status}}

Summary of relevant context from {{name}}'s memory: {{context}}

### Input:
The {{user}} say "{{question}}". What should {{name}} response?

### Response:
Here is the response from {{name}}: "{{gen 'response' temperature=0.95 stop='"'}}\""""