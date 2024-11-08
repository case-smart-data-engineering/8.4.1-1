import itertools
import codecs
import os
from itertools import combinations
from collections import defaultdict, deque
import logging

# 字典用于存储实体和关系的ID映射
entity2id = {}
relation2id = {}


# 加载数据集
# 返回实体集合，关系集合，三元组列表。（其中实体集合、关系集合和三元组列表的元素都是实体和关系对应的id）
def data_loader(file):
    file1 = os.path.join(file, "data/train.txt")
    file2 = os.path.join(file, "data/entity2id.txt")
    file3 = os.path.join(file, "data/relation2id.txt")

    # 读取实体和关系文件，把实体id文档和关系id文档中的内容装入实体id和关系id字典中，字典结构为{实体:id}
    with open(file2, 'r') as f1, open(file3, 'r') as f2:
        # 读取整个文件所有行，保存在列表lines1和lines2中，每行作为列表的一个元素，类型为str,结构为'实体 \t id'
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        for line in lines1:
            # 按照'\t'符对lines1的每个元素line进行分割，操作之后line变成了一个字符串列表,结构为['实体','id']
            line = line.strip().split('\t')
            if len(line) != 2:
                # 进行下一轮循环
                continue
                # 列表line的实体和id分别作为实体id字典的key和value
            entity2id[line[0]] = line[1]

        for line in lines2:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            relation2id[line[0]] = line[1]

    entity_set = set()
    relation_set = set()
    triple_list = []

    # 读取训练集文件，
    with codecs.open(file1, 'r') as f:
        # 读取训练集文件的所有行，保存在列表content中，每行作为列表的一个元素，类型为str,结构为'头实体 \t 尾实体 \t 关系'
        content = f.readlines()
        for line in content:
            # 按照'\t'符对content的每个元素line进行分割，操作之后变成了一个字符串列表triple,结构为['头实体','尾实体','关系']
            triple = line.strip().split("\t")
            if len(triple) != 3:
                continue
            # 把实体和关系在id字典中对应的value赋给h_,t_,r_，且这三个值都是id
            h_ = entity2id[triple[0]]
            t_ = entity2id[triple[1]]
            r_ = relation2id[triple[2]]
            # h_,t_,r_组成一个列表，装入triple_list
            triple_list.append([h_, t_, r_])

            entity_set.add(h_)
            entity_set.add(t_)

            relation_set.add(r_)

    return entity_set, relation_set, triple_list

logging.basicConfig(level=logging.INFO)

# 定义Atom类，代表知识库中的一个原子（事实）
class Atom:
    def __init__(self, relation, args):
        self.relation = relation
        self.args = args

    def __repr__(self):
        return f"{self.relation}({self.args[0]},{self.args[1]})"

class InMemoryDatabase:
    def __init__(self):
        self.facts = defaultdict(set)

    # 向数据库中添加一个事实
    # relation是关系名，args是一个包含(头实体,尾实体)的元组
    def add_fact(self, relation, args):
        self.facts[relation].add(tuple(args))

    def get_facts(self, relation, args=None):
        if args is None:
            return self.facts[relation]
        else:
            return {fact for fact in self.facts[relation] if set(fact) == set(args)}

# 定义一个Rule类，代表一个推理规则
class Rule:
    def __init__(self, body, head):
        self.body = body
        self.head = head

    def __repr__(self):
        body_str = ', '.join([f"{atom}" for atom in self.body])
        head_str = f"{self.head}"
        return f"{body_str} => {head_str}"

    def __hash__(self):
        return hash((tuple(self.body), self.head))

    def __eq__(self, other):
        # 返回Rule格式为："body => head"
        return self.body == other.body and self.head == other.head

class AMIE:
    def __init__(self, kb, min_confidence=0.9, min_head_coverage=0.9, max_depth=3):
        self.kb = kb
        self.min_confidence = min_confidence  # 最小置信度阈值
        self.min_head_coverage = min_head_coverage  # 最小头部覆盖率阈值
        self.max_depth = max_depth  # 规则生成时的最大深度
        self.rules = set()

    # 计算规则置信度
    def calculate_confidence(self, rule):
        # (body)实例数
        body_facts_list = [self.kb.get_facts(atom.relation) for atom in rule.body]
        if len(body_facts_list) == 0:
            return 0
        body_facts = set.intersection(*body_facts_list)

        # (body -> head)实例数
        head_facts = self.kb.get_facts(rule.head.relation)
        num_body_head = len([fact for fact in head_facts if all(fact in body_facts for atom in rule.body)])

        # 计算confidence
        if len(body_facts) == 0:
            return 0
        confidence = num_body_head / len(body_facts)

        return confidence

    # 根据置信度和头部覆盖率对规则进行剪枝
    def pruning(self, rule):
        # 如果规则已经存在于规则集中，则不进行添加
        if rule in self.rules:
            return False

        # 计算头部覆盖率，即头部关系对应的事实数量与总事实数量的比例
        head_facts = self.kb.get_facts(rule.head.relation)
        head_coverage = len(head_facts) / (len(self.kb.facts[rule.head.relation]) + 1e-9)

        # 计算规则的置信度
        confidence = self.calculate_confidence(rule)

        if head_coverage < self.min_head_coverage or confidence < self.min_confidence:
            return False
        self.rules.add(rule)
        return True

    # 通过向规则体中添加“悬挂”原子来生成新规则
    def add_dangling_atom(self, rule, depth):
        if depth >= self.max_depth:
            return set()

        new_rules = set()
        existing_vars = {arg for atom in rule.body for arg in atom.args}

        for atom in rule.body:
            for fact in self.kb.get_facts(atom.relation):
                for new_var in fact:
                    if new_var not in existing_vars:
                        new_atom = Atom(atom.relation, (atom.args[0], new_var))
                        new_body = list(rule.body) + [new_atom]
                        new_rule = Rule(new_body, rule.head)

                        if self.pruning(new_rule) and new_rule not in self.rules:
                            new_rules.add(new_rule)
                            new_rules.update(self.add_dangling_atom(new_rule, depth + 1))

        return new_rules

    # 通过实例化规则体中的原子生成新规则
    def add_instantiated_atom(self, rule, depth):
        if depth >= self.max_depth:
            return set()
        new_rules = set()
        seen_pairs = set()
        for atom in rule.body:
            for entity in self.kb.get_facts(atom.relation):
                if entity[0] == atom.args[0]:
                    new_pair = (entity[1], atom.args[1])
                    if new_pair not in seen_pairs:
                        seen_pairs.add(new_pair)
                        new_atom = Atom(atom.relation, new_pair)
                        new_body = rule.body + [new_atom]
                        new_rule = Rule(new_body, rule.head)
                        if self.pruning(new_rule):
                            new_rules.add(new_rule)
        return new_rules

    # 通过向规则体中添加“闭合”原子来生成新规则
    def add_closed_atom(self, rule, depth):
        if depth >= self.max_depth:
            return set()
        new_rules = set()
        for atom1, atom2 in itertools.combinations(rule.body, 2):
            if atom1.relation == atom2.relation and atom1.args[0] == atom2.args[1]:
                new_atom = Atom(atom1.relation, (atom1.args[1], atom2.args[0]))
                new_body = rule.body + [new_atom]
                new_rule = Rule(new_body, rule.head)
                if self.pruning(new_rule):
                    new_rules.add(new_rule)
        return new_rules

    def run(self):
        initial_rules = self.generate_initial_rules()
        to_process = deque([(rule, 0) for rule in initial_rules])
        processed = set()
        while to_process:
            rule, depth = to_process.popleft()
            if rule not in processed and self.pruning(rule):
                processed.add(rule)
                to_process.extend([(new_rule, depth + 1) for new_rule in self.add_dangling_atom(rule, depth)])
                to_process.extend([(new_rule, depth + 1) for new_rule in self.add_instantiated_atom(rule, depth)])
                to_process.extend([(new_rule, depth + 1) for new_rule in self.add_closed_atom(rule, depth)])
        for rule in self.rules:
            logging.info(rule)

    def generate_initial_rules(self):
        initial_rules = set()
        relations = list(self.kb.facts.keys())
        for head_relation in relations:
            for fact in self.kb.get_facts(head_relation):
                head = Atom(head_relation, fact)
                # 生成多个不同关系的body，数量最少2个
                body_relations = [rel for rel in relations if rel != head_relation]
                for body_relation_comb in combinations(body_relations, 2):
                    body = [Atom(body_relation, body_fact) for body_relation in body_relation_comb for body_fact in
                            self.kb.get_facts(body_relation)]
                    initial_rules.add(Rule(body, head))
        return initial_rules

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    directory = os.getcwd()
    entity_set, relation_set, triple_list = data_loader(directory)
    kb = InMemoryDatabase()
    for triple in triple_list:
        h, r, t = triple
        kb.add_fact(r, (h, t))

    amie = AMIE(kb)
    amie.run()
