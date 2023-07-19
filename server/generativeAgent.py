from datetime import datetime, timedelta
from langchain.docstore import InMemoryDocstore
# from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
from langchain.schema import Document

import math
import faiss
from .prompt import *
from .time_weighted_retriever import TimeWeightedVectorStoreRetrieverModified

import numpy as np

def score_normalizer(val: float) -> float:
    return 1 - 1 / (1 + np.exp(val))

# def score_normalizer(val: float) -> float:
#     return val

def get_text_from_docs(list_docs, include_time = False):
    texts = ""
    for i, doc in enumerate(list_docs):
        if include_time:
            time_t = doc.metadata['created_at'].strftime('%A %B %d, %Y, %H:%M') + ": "            
        else:
            time_t = ""
        if i == 0:
            texts += "- " + time_t + doc.page_content
        else:
            texts += "\n- " + time_t + doc.page_content
    return texts

def merge_docs(docs1, docs2):
    list_index1 = []
    docs_merged = []
    for doc_t in docs1:
        list_index1.append(doc_t.metadata['buffer_idx'])
        docs_merged.append(doc_t)
    for doc_t in docs2:
        if not (doc_t.metadata['buffer_idx'] in list_index1):
            docs_merged.append(doc_t)
    return docs_merged

# Based on 
# https://github.com/hwchase17/langchain/blob/master/langchain/experimental/generative_agents/generative_agent.py
class GenerativeAgent:
    def __init__(self, guidance, name, age, des, trails, embeddings_model, current_time=None):
        self.guidance = guidance
        self.name = name
        self.age = str(age)
        self.des = des.split(';')
        self.trails = trails
        self.summary = trails
        self.plan = []
        self.status = None        
        embedding_size = 384
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {}, relevance_score_fn=score_normalizer)
        self.retriever = TimeWeightedVectorStoreRetrieverModified(vectorstore=vectorstore, other_score_keys=["importance"], k=10, decay_rate=0.01)
        self.current_time = current_time
        if self.current_time is None:
            self.last_refreshed = datetime.now()
        else:
            self.last_refreshed = current_time
        self.summary_refresh_seconds = 3600
        self.aggregate_importance = 0
        self.reflecting = False
        self.reflection_threshold = 25
        self.dialogue_list = []
        self.relevant_memories = ''
        self.silent = False

        self.add_memories(self.des)
        
    def set_current_time(self, time):
        self.current_time = time

    def get_current_time(self,):
        if self.current_time is not None:
            return self.current_time
        else:
            return datetime.now()

    def next_task(self,):
        self.set_current_time(self.status['to'])
        return self.update_status()

    def update_status(self,):
        current_time = self.get_current_time()
        need_replan = True
        for task_temp in self.plan:
            # task_to_temp = datetime.strptime(task_temp['to'], '%H:%M')
            task_to_temp = task_temp['to']
            if task_to_temp > current_time:
                self.status = task_temp
                need_replan = False
                break
        if need_replan:
            new_plan = self.make_plan()
            self.status = new_plan[0]
        return self.status
        
    def add_memories(self, list_mem):
        for mem_temp in list_mem:
            if isinstance(mem_temp, dict):
                mem_des, mem_time = mem_temp
            else:
                mem_des = mem_temp
                mem_time = self.get_current_time()
            
            prompt = self.guidance(PROMPT_ADDMEM, silent=self.silent)
            result = prompt(mem=mem_des)
            # importance_score_temp = int(result['rate'])*self.importance_weight
            importance_score_temp = int(result['rate'])
            self.retriever.add_documents([Document(page_content=mem_des, metadata={"importance": importance_score_temp, "created_at": mem_time})], current_time=mem_time)
            self.aggregate_importance += int(result['rate'])

            

        if not self.reflecting and self.aggregate_importance > self.reflection_threshold:
            self.reflecting = True
            self._relection()
            self.aggregate_importance = 0.0
            self.reflecting = False

    def _get_salient(self,):
        # number of recent memories
        last_k = 20
        recent_memories_list = self.retriever.memory_stream[-last_k:]
        recent_memories_text = get_text_from_docs(recent_memories_list, include_time = True)
        
        prompt = self.guidance(PROMPT_SALIENT, silent=self.silent)
        result = prompt(recent_memories=recent_memories_text)
        return result['items']

    def _get_insights(self, list_docs):
        docs = list_docs
        statements = get_text_from_docs(docs, include_time = False)
        prompt = self.guidance(PROMPT_INSIGHTS, silent=self.silent)
        result = prompt(statements=statements)
        return result['items']

    def _relection(self,):
        list_salient = self._get_salient()
        list_docs = []
        for salient_temp in list_salient:
            docs = self.retriever.get_relevant_documents(salient_temp, self.get_current_time())
            list_docs = merge_docs(list_docs, docs)
        list_insights = self._get_insights(list_docs)
        self.add_memories(list_insights)

    def get_summary(self, force_refresh=False, now=None):
        current_time = self.get_current_time() if now is None else now
        since_refresh = (current_time - self.last_refreshed).seconds

        if (
            not self.summary
            or since_refresh >= self.summary_refresh_seconds
            or force_refresh
        ):
            core_characteristics = self._run_characteristics()
            daily_occupation = self._run_occupation()
            feeling = self._run_feeling()

            description = core_characteristics + '. ' + daily_occupation + '. ' + feeling            
            self.summary = (f"Name: {self.name} (age: {self.age})" + f"\nTrails: {self.trails}" + f"\nSummary: {description}")
            self.last_refreshed = current_time            
        return self.summary
        
    def _run_characteristics(self,):
        docs = self.retriever.get_relevant_documents(self.name + "'s core characteristics", self.get_current_time())
        statements = get_text_from_docs(docs, include_time = False)

        prompt = self.guidance(PROMPT_CHARACTERISTICS, silent=self.silent)
        result = prompt(statements=statements, name=self.name)
        return result['res']

    def _run_occupation(self,):
        docs = self.retriever.get_relevant_documents(self.name + "'s current daily occupation", self.get_current_time())
        statements = get_text_from_docs(docs, include_time = False)

        prompt = self.guidance(PROMPT_OCCUPATION, silent=self.silent)
        result = prompt(statements=statements, name=self.name)
        return result['res']

    def _run_feeling(self,):
        docs = self.retriever.get_relevant_documents(self.name + "'s feeling about his recent progress in life", self.get_current_time())
        statements = get_text_from_docs(docs, include_time = False)

        prompt = self.guidance(PROMPT_FEELING, silent=self.silent)
        result = prompt(statements=statements, name=self.name)
        return result['res']

    def make_plan(self,):
        now = self.get_current_time().strftime('%H:%M')
        prompt = self.guidance(PROMPT_PLAN, silent=self.silent)
        result = prompt(summary=self.summary,
                        name=self.name,
                        now=now,
                        current_time=self.get_current_time().strftime('%A %B %d, %Y, %H:%M')
                       )

        current_date = self.get_current_time()
        list_task = result['items']
        list_task.insert(0, {'from': now, 'to': result['to'], 'task': result['task']})
        list_task_time = []
        for i, task_temp in enumerate(list_task):
            t_from = datetime.strptime(task_temp['from'], '%H:%M')
            t_from = current_date.replace(hour=t_from.hour, minute=t_from.minute)
            t_to = datetime.strptime(task_temp['to'], '%H:%M')
            t_to = current_date.replace(hour=t_to.hour, minute=t_to.minute)
            delta_time = (t_to - t_from)
            if delta_time.total_seconds() < 0:
                t_to += timedelta(days=1)
            list_task_time.append({'from': t_from, 'to': t_to, 'task': task_temp['task']})
            
        self.plan = list_task_time
        return list_task_time

    def react(self, observation, observed_entity, entity_status):
        self.add_memories([observation])
        if isinstance(observed_entity, str):
            name_observed_entity = observed_entity
        else:
            name_observed_entity = observed_entity.name
            
        bool_react, reaction, context = self._check_reaction(observation, name_observed_entity, entity_status)
        if bool_react == 'Yes':
            if isinstance(observed_entity, GenerativeAgent):
                self._start_dialogue(observation, name_observed_entity, entity_status, context, reaction)
            new_plan = self._replan(observation, reaction)
            self.plan = new_plan
            self.update_status()
        return bool_react, reaction, context

    def _start_dialogue(self, observation, name_observed_entity, entity_status, context, reaction):
        prompt = self.guidance(PROMPT_DIALOGUE, silent=self.silent)
        result = prompt(summary=self.summary,
                        name=self.name,
                        status=self.status['task'],
                        observation=observation,
                        reaction=reaction,
                        observed_entity=name_observed_entity,
                        context=context,
                        current_time=self.get_current_time().strftime('%A %B %d, %Y, %H:%M')
                       )
        self.dialogue_list.append(f"{self.get_current_time().strftime('%A %B %d, %Y, %H:%M')}\n{result['dialogue']}")
        return result['dialogue']
        
    def _get_relevant_context(self, observed_entity, entity_status):
        docs1 = self.retriever.get_relevant_documents(f"What is {self.name}'s relationship with {observed_entity}?", self.get_current_time())        
        docs2 = self.retriever.get_relevant_documents(entity_status, self.get_current_time())
        
        docs = merge_docs(docs1, docs2)
        statements = get_text_from_docs(docs, include_time = False)
        self.relevant_memories = statements
        prompt = self.guidance(PROMPT_CONTEXT, silent=self.silent)
        result = prompt(statements=statements, name=self.name, observed_entity=observed_entity, entity_status=entity_status)
        return result['context']

    def _check_reaction(self, observation, observed_entity, entity_status):
        context = self._get_relevant_context(observed_entity, entity_status)
        prompt = self.guidance(PROMPT_REACT, silent=self.silent)
        result = prompt(summary=self.summary,
                        name=self.name,
                        status=self.status['task'],
                        observation=observation,
                        observed_entity=observed_entity,
                        context=context,
                        current_time=self.get_current_time().strftime('%A %B %d, %Y, %H:%M'),
                        valid_opts=['Yes', 'No']
                       )
        return result['reaction'], result['result'], context

    def _replan(self, observation, reaction):
        now = self.get_current_time().strftime('%H:%M')
        prompt = self.guidance(PROMPT_REPLAN, silent=self.silent)
        result = prompt(summary=self.summary,
                        name=self.name,
                        status=self.status['task'],
                        observation=observation,
                        reaction=reaction,
                        now=now,
                        current_time=self.get_current_time().strftime('%A %B %d, %Y, %H:%M')
                       )
        list_task = result['items']
        list_task.insert(0, {'from': now, 'to': result['to'], 'task': result['task']})

        current_date = self.get_current_time()
        list_task_time = []
        for i, task_temp in enumerate(list_task):
            t_from = datetime.strptime(task_temp['from'], '%H:%M')
            t_from = current_date.replace(hour=t_from.hour, minute=t_from.minute)
            t_to = datetime.strptime(task_temp['to'], '%H:%M')
            t_to = current_date.replace(hour=t_to.hour, minute=t_to.minute)
            delta_time = (t_to - t_from)
            if delta_time.total_seconds() < 0:
                t_to += timedelta(days=1)
            list_task_time.append({'from': t_from, 'to': t_to, 'task': task_temp['task']})
        return list_task_time
        
    def interview(self, user, question):
        # context = self._get_relevant_context(user, question)
        docs = self.retriever.get_relevant_documents(question, self.get_current_time())      
        context = get_text_from_docs(docs, include_time = False)
        self.relevant_memories = context
        
        prompt = self.guidance(PROMPT_INTERVIEW, silent=self.silent)
        result = prompt(summary=self.summary,
                        name=self.name,
                        status=self.status['task'],
                        user=user,
                        context=context,
                        question=question,
                        current_time=self.get_current_time().strftime('%A %B %d, %Y, %H:%M')
                       )
        return result['response']
