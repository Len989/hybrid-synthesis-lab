"""
Hybrid Synthesis Laboratory — с AI-интерпретацией и визуализацией графа
"""

import streamlit as st
from dataclasses import dataclass, field
from collections import defaultdict
import itertools
from typing import List, Tuple, Dict, Set, Optional, Callable
import json
from datetime import datetime
import requests
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import math
import random
from enum import Enum
import pandas as pd


class ReductionStrategy(Enum):
    """Стратегии применения правил переписывания."""
    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"
    RANDOM = "random"
    SHORTEST_FIRST = "shortest_first"
    LONGEST_FIRST = "longest_first"

# ═══════════════════════════════════════════════════════════════════
# CORE ENGINE (Term, CongruenceClosure — без изменений)
# ═══════════════════════════════════════════════════════════════════

class Term:
    """Символьный терм для алгебраических выражений."""

    def __init__(self, head: str, args: List["Term"] | None = None, provenance: str = ""):
        self.head = head
        self.args = args or []
        self.provenance = provenance  # для отладки и улучшения отображения (не влияет на == и hash)

    def __repr__(self):
        if not self.args:
            return self.head
        return f"{self.head}({', '.join(map(repr, self.args))})"

    def __eq__(self, other):
        if not isinstance(other, Term):
            return False
        return self.head == other.head and self.args == other.args

    def __hash__(self):
        return hash((self.head, tuple(self.args)))

    def substitute(self, mapping: Dict[str, "Term"]) -> "Term":
        if not self.args:
            return mapping.get(self.head, self)
        return Term(self.head, [arg.substitute(mapping) for arg in self.args])

    def variables(self) -> Set[str]:
        if not self.args:
            return {self.head} if self.head[0].islower() else set()
        result = set()
        for arg in self.args:
            result.update(arg.variables())
        return result

    def to_dict(self) -> dict:
        if not self.args:
            return {"head": self.head, "args": []}
        return {"head": self.head, "args": [arg.to_dict() for arg in self.args]}

    @classmethod
    def from_dict(cls, d: dict) -> "Term":
        return cls(d["head"], [cls.from_dict(arg) for arg in d["args"]])


class CongruenceClosure:
    """Замыкание конгруэнции: Union-Find с безопасным распространением."""

    def __init__(self):
        self.parent: Dict[Term, Term] = {}
        self.rank: Dict[Term, int] = {}

    def find(self, t: Term) -> Term:
        if t not in self.parent:
            self.parent[t] = t
            self.rank[t] = 0
            return t
        if self.parent[t] != t:
            self.parent[t] = self.find(self.parent[t])
        return self.parent[t]

    def union(self, t1: Term, t2: Term) -> bool:
        p1 = self.find(t1)
        p2 = self.find(t2)
        if p1 == p2:
            return False
        if self.rank[p1] < self.rank[p2]:
            self.parent[p1] = p2
        elif self.rank[p1] > self.rank[p2]:
            self.parent[p2] = p1
        else:
            self.parent[p2] = p1
            self.rank[p1] += 1
        return True

    def close(self, equations: List[Tuple[Term, Term]], arities: Dict[str, int]):
        """Замыкание: начальное объединение + ограниченное распространение."""
        for left, right in equations:
            self.union(left, right)

        changed = True
        max_iterations = 50
        iteration = 0

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            all_terms = list(self.parent.keys())

            for op, arity in arities.items():
                if arity == 0:
                    continue
                for t in all_terms:
                    if t.head != op or len(t.args) != arity:
                        continue
                    for i in range(arity):
                        root_i = self.find(t.args[i])
                        for other in all_terms:
                            if self.find(other) == root_i and other != t.args[i]:
                                new_args = list(t.args)
                                new_args[i] = other
                                new_t = Term(op, new_args)
                                if self.union(t, new_t):
                                    changed = True
                                    break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break



def parse_term_string(s: str) -> Optional[Term]:
        """Парсит строку вида 'op(arg1, arg2)' или 'const' в Term."""
        s = s.strip()
        if not s:
            return None
    
        # Если есть скобки — парсим как функцию
        if '(' in s and s.endswith(')'):
            paren_idx = s.index('(')
            head = s[:paren_idx].strip()
            args_str = s[paren_idx+1:-1].strip()
        
            # Разбираем аргументы (с учётом вложенных скобок)
            args = []
            depth = 0
            current = ""
            for ch in args_str:
                if ch == '(':
                    depth += 1
                    current += ch
                elif ch == ')':
                    depth -= 1
                    current += ch
                elif ch == ',' and depth == 0:
                    arg = parse_term_string(current)
                    if arg:
                        args.append(arg)
                    current = ""
                else:
                    current += ch
            if current.strip():
                arg = parse_term_string(current)
                if arg:
                    args.append(arg)
        
            return Term(head, args)
        else:
            # Константа или переменная
            return Term(s)
            
# ═══════════════════════════════════════════════════════════════════
# ATOM DATA STRUCTURE
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Atom:
    name: str
    carrier: List[str]
    operations: Dict[str, int]
    axioms: List[Tuple[Term, Term]] = field(default_factory=list)
    description: str = ""
    is_synthetic: bool = False
    parent_atoms: List[str] = field(default_factory=list)
    interaction: str = ""
    synthesis_date: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "carrier": self.carrier,
            "operations": self.operations,
            "axioms": [(left.to_dict(), right.to_dict()) for left, right in self.axioms],
            "description": self.description,
            "is_synthetic": self.is_synthetic,
            "parent_atoms": self.parent_atoms,
            "interaction": self.interaction,
            "synthesis_date": self.synthesis_date
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Atom":
        axioms = [(Term.from_dict(left), Term.from_dict(right)) for left, right in d.get("axioms", [])]
        return cls(
            name=d["name"],
            carrier=d["carrier"],
            operations=d["operations"],
            axioms=axioms,
            description=d.get("description", ""),
            is_synthetic=d.get("is_synthetic", False),
            parent_atoms=d.get("parent_atoms", []),
            interaction=d.get("interaction", ""),
            synthesis_date=d.get("synthesis_date", "")
        )


# ═══════════════════════════════════════════════════════════════════
# SYNTHESIS ENGINE
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SynthesisResult:
    atom: Optional[Atom]
    collapsed: bool
    classes: Dict[Term, List[Term]]
    equations_count: int
    timestamp: str
    cc: Optional[CongruenceClosure] = None
    parent_A: Optional[Atom] = None
    parent_B: Optional[Atom] = None
    action_name: str = ""


def synthesize(A: Atom, B: Atom, action_name: str = "·", 
               user_equations: List[Tuple[str, str]] = None,
               custom_equations: List[Tuple[str, str]] = None,
               aggressive_normalization: bool = False,
               shadow_analysis: bool = False) -> SynthesisResult:
    all_ops = {}
    all_ops.update(A.operations)
    all_ops.update(B.operations)
    all_ops[action_name] = 2

    equations: List[Tuple[Term, Term]] = [] 
                   
    # 0. Пользовательские отождествления (контекст)
    if user_equations:
        for left_str, right_str in user_equations:
            equations.append((Term(left_str), Term(right_str)))
    # 0.5 Кастомные равенства термов
    if custom_equations:
        for left_str, right_str in custom_equations:
            left_term = parse_term_string(left_str)
            right_term = parse_term_string(right_str)
            if left_term is not None and right_term is not None:
                equations.append((left_term, right_term))
                
    # 1. Аксиомы A
    for left, right in A.axioms:
        vars_left = left.variables()
        vars_right = right.variables()
        all_vars = vars_left | vars_right

        if len(all_vars) == 0:
            equations.append((left, right))
        elif len(all_vars) <= 2:
            for combo in itertools.product(A.carrier, repeat=len(all_vars)):
                mapping = {var: Term(val) for var, val in zip(sorted(all_vars), combo)}
                equations.append((left.substitute(mapping), right.substitute(mapping)))

    # 2. Перенесённые аксиомы B
    for left_b, right_b in B.axioms:
        vars_left = left_b.variables()
        vars_right = right_b.variables()
        all_vars_b = vars_left | vars_right

        if len(all_vars_b) == 0:
            for a in A.carrier:
                equations.append((
                    Term(action_name, [left_b, Term(a)]),
                    Term(action_name, [right_b, Term(a)])
                ))
        elif len(all_vars_b) <= 2:
            for combo in itertools.product(B.carrier, repeat=len(all_vars_b)):
                mapping_b = {var: Term(val) for var, val in zip(sorted(all_vars_b), combo)}
                left_sub = left_b.substitute(mapping_b)
                right_sub = right_b.substitute(mapping_b)
                for a in A.carrier:
                    a_term = Term(a)
                    equations.append((
                        Term(action_name, [left_sub, a_term]),
                        Term(action_name, [right_sub, a_term])
                    ))

    # 3. Совместимость действия с операциями A
    for op, arity in A.operations.items():
        if arity == 0:
            continue
        for b in B.carrier:
            b_term = Term(b)
            for combo in itertools.product(A.carrier, repeat=arity):
                args = [Term(x) for x in combo]
                left = Term(action_name, [b_term, Term(op, args)])
                right = Term(op, [Term(action_name, [b_term, arg]) for arg in args])
                equations.append((left, right))

    # Вычисление коуравнителя
    cc = CongruenceClosure()
    cc.close(equations, all_ops)

    # Сбор классов
    classes = defaultdict(list)
    for t in list(cc.parent.keys()):
        classes[cc.find(t)].append(t)

    # ── НОРМАЛИЗАЦИЯ ЧЕРЕЗ REWRITING (ФИНАЛЬНАЯ) ───────
    rs = build_rewriting_system(A, action_name)

    # Если включена агрессивная нормализация — добавляем кастомные равенства как правила редукции
    if aggressive_normalization and custom_equations:
        for left_str, right_str in custom_equations:
            left_term = parse_term_string(left_str)
            right_term = parse_term_string(right_str)
            if left_term is not None and right_term is not None:
                rs.add_rule(left_term, right_term)

    term_to_normal = {}
    for t in list(cc.parent.keys()):
        term_to_normal[t] = rs.normalize(t)

    raw_classes = defaultdict(list)
    for t, norm_t in term_to_normal.items():
        raw_classes[norm_t].append(t)

    final_classes = defaultdict(set)
    visited = set()
    for norm_rep, elems in raw_classes.items():
        if norm_rep in visited:
            continue
        root = cc.find(norm_rep) if norm_rep in cc.parent else norm_rep
        for other_norm, other_elems in raw_classes.items():
            other_root = cc.find(other_norm) if other_norm in cc.parent else other_norm
            if cc.find(root) == cc.find(other_root):
                for e in other_elems:
                    final_classes[root].add(e)
                visited.add(other_norm)
        visited.add(norm_rep)

    final_classes_dict = {}
    basic_terms = {Term(el) for el in A.carrier}
    for op, arity in A.operations.items():
        if arity == 0:
            basic_terms.add(Term(op, []))

    for root, elems in final_classes.items():
        best = None
        for e in elems:
            if e in basic_terms:
                best = e
                break
        if best is None:
            best = min(elems, key=lambda x: len(repr(rs.normalize(x))))
        final_classes_dict[best] = sorted(elems, key=lambda x: len(repr(x)))

    classes = final_classes_dict
    # ── КОНЕЦ НОРМАЛИЗАЦИИ ─────────────────────────────

    # ── СКЛЕЙКА ДУБЛИКАТОВ ПОСЛЕ НОРМАЛИЗАЦИИ ─────────
    merged_classes = {}
    for rep, elems in classes.items():
        if rep in merged_classes:
            for e in elems:
                if e not in merged_classes[rep]:
                    merged_classes[rep].append(e)
        else:
            merged_classes[rep] = elems[:]
    classes = merged_classes
    # ── КОНЕЦ СКЛЕЙКИ ДУБЛИКАТОВ ───────────────────────

    # ── ФИНАЛЬНАЯ НОРМАЛИЗАЦИЯ: ЗАМЕНА СЛОЖНЫХ ТЕРМОВ НА БАЗОВЫЕ ─
    basic_terms = {Term(el) for el in A.carrier}
    for op, arity in A.operations.items():
        if arity == 0:
            basic_terms.add(Term(op, []))

    simplification_rules = {}
    for rep, elems in classes.items():
        best = None
        for e in elems:
            if e in basic_terms:
                best = e
                break
        if best is not None:
            for e in elems:
                if e != best:
                    simplification_rules[e] = best

    simplified_classes = {}
    for rep, elems in classes.items():
        new_rep = simplification_rules.get(rep, rep)
        new_elems = []
        seen = set()
        for e in elems:
            new_e = simplification_rules.get(e, e)
            if new_e not in seen:
                new_elems.append(new_e)
                seen.add(new_e)
        if new_rep in simplified_classes:
            for e in new_elems:
                if e not in simplified_classes[new_rep]:
                    simplified_classes[new_rep].append(e)
        else:
            simplified_classes[new_rep] = new_elems

    classes = simplified_classes
    # ── КОНЕЦ ФИНАЛЬНОЙ НОРМАЛИЗАЦИИ ──────────────────────────

    # Ещё один проход нормализации для всех представителей
    for rep in list(classes.keys()):
        norm_rep = rs.normalize(rep)
        if norm_rep != rep:
            if norm_rep in classes:
                for e in classes[rep]:
                    if e not in classes[norm_rep]:
                        classes[norm_rep].append(e)
            else:
                classes[norm_rep] = classes[rep]
            del classes[rep]

    # === ПРАВИЛЬНОЕ ДОБАВЛЕНИЕ ACTION-ТЕРМОВ (с provenance) ===
    if action_name and B is not None:
        for b_elem in B.carrier:
            for a_elem in A.carrier:
                action_term = Term(action_name, [Term(b_elem), Term(a_elem)], provenance="action")
                norm_action = rs.normalize(action_term)

                # Добавляем в CongruenceClosure
                if cc is not None:
                    if norm_action not in cc.parent:
                        cc.parent[norm_action] = norm_action
                        cc.rank[norm_action] = 0

                # Добавляем в classes (даже если нормализовался в существующий)
                placed = False
                for rep in list(classes.keys()):
                    if norm_action == rep or norm_action in classes[rep]:
                        if norm_action not in classes[rep]:
                            classes[rep].append(norm_action)
                        placed = True
                        break
                if not placed:
                    classes[norm_action] = [norm_action]

    # Проверка коллапса
    carrier_terms = [rs.normalize(Term(el)) for el in A.carrier]
    distinct_roots = {cc.find(t) for t in carrier_terms}
    collapsed = len(distinct_roots) <= 1

    if collapsed:
        return SynthesisResult(
            atom=None,
            collapsed=True,
            classes=dict(classes),
            equations_count=len(equations),
            timestamp=datetime.now().isoformat(),
            cc=cc,
            parent_A=A,
            parent_B=B,
            action_name=action_name
        )
        # ── Shadow analysis (если включено, вычисляем до возврата) ──
    if shadow_analysis:
        # Метрики будут вычислены в UI, но сохраняем флаг в результате
        pass

                   

    # Построение нового атома
    new_carrier = []
    carrier_repr_map = {}
    for t in carrier_terms:
        norm_t = rs.normalize(t)
        if norm_t not in carrier_repr_map:
            repr_name = repr(norm_t)
            new_carrier.append(repr_name)
            carrier_repr_map[norm_t] = repr_name

    new_operations = {}
    # Сохраняем ВСЕ операции атома A (и константы, и бинарные)
    for op, arity in A.operations.items():
        new_operations[op] = arity

    new_operations[action_name] = 2

    # Гарантированно добавляем константу 1, если она выжила в носителе
    if "1" in new_carrier and "1" not in new_operations:
        new_operations["1"] = 0

    new_atom = Atom(
        name=f"{A.name}⊕{B.name}",
        carrier=new_carrier,
        operations=new_operations,
        axioms=[],
        description=f"Гибрид {A.name} и {B.name} через {action_name}",
        is_synthetic=True,
        parent_atoms=[A.name, B.name],
        interaction=action_name,
        synthesis_date=datetime.now().isoformat()
    )

    return SynthesisResult(
        atom=new_atom,
        collapsed=False,
        classes=dict(classes),
        equations_count=len(equations),
        timestamp=datetime.now().isoformat(),
        cc=cc,
        parent_A=A,
        parent_B=B,
        action_name=action_name
    )


# ═══════════════════════════════════════════════════════════════════
# GRAPH VISUALIZER
# ═══════════════════════════════════════════════════════════════════

def build_synthesis_graph(result: SynthesisResult) -> plt.Figure:
    """Строит граф структуры: узлы = классы эквивалентности, рёбра = операции."""
    G = nx.Graph()

    classes = result.classes
    A = result.parent_A
    B = result.parent_B
    action_name = result.action_name

    A_carrier_set = set(A.carrier) if A else set()
    B_carrier_set = set(B.carrier) if B else set()

    node_colors = []
    node_sizes = []
    node_labels = {}

    for rep, elems in classes.items():
        rep_str = repr(rep)
        G.add_node(rep_str)
        node_labels[rep_str] = rep_str

        # Размер узла
        size = 300.0 + float(len(elems)) * 10.0
        if size > 2000.0:
            size = 2000.0
        node_sizes.append(size)

        # Цвет
        rep_head = rep.head if isinstance(rep, Term) else rep_str
        in_A = any(rep_head == el for el in A_carrier_set) or (
            isinstance(rep, Term) and hasattr(rep, 'head') and rep.head in A.operations
        )
        in_B = any(rep_head == el for el in B_carrier_set) or (
            isinstance(rep, Term) and hasattr(rep, 'head') and rep.head in B.operations
        )

        if in_A and not in_B:
            node_colors.append('#3498db')
        elif in_B and not in_A:
            node_colors.append('#2ecc71')
        elif in_A and in_B:
            node_colors.append('#9b59b6')
        else:
            node_colors.append('#e74c3c')

    # Рёбра
    edge_colors = []
    edge_styles = []

    if A and result.cc:
        # Операции A
        for op_name, arity in A.operations.items():
            if arity != 2:
                continue
            for a1 in A.carrier:
                for a2 in A.carrier:
                    term = Term(op_name, [Term(a1), Term(a2)])
                    root = result.cc.find(term) if term in result.cc.parent else term
                    rep_str = None
                    for rep, elems in classes.items():
                        if rep == root or root in elems:
                            rep_str = repr(rep)
                            break
                    if rep_str and repr(Term(a1)) != rep_str:
                        G.add_edge(repr(Term(a1)), rep_str)
                        edge_colors.append('#555555')
                        edge_styles.append('solid')

        # Действие
        if action_name and B:
            for b_elem in B.carrier:
                for a_elem in A.carrier:
                    action_term = Term(action_name, [Term(b_elem), Term(a_elem)])
                    root = result.cc.find(action_term) if action_term in result.cc.parent else action_term
                    rep_str = None
                    for rep, elems in classes.items():
                        if rep == root or root in elems:
                            rep_str = repr(rep)
                            break
                    if rep_str and repr(Term(a_elem)) != rep_str:
                        G.add_edge(repr(Term(a_elem)), rep_str)
                        edge_colors.append('#e67e22')
                        edge_styles.append('dotted')

    # Позиционирование
    if len(G.nodes) == 0:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0e1117')
        ax.text(0.5, 0.5, 'Пустой граф', color='white', ha='center', va='center')
        ax.set_facecolor('#0e1117')
        ax.axis('off')
        return fig

    pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0e1117')

    # Рёбра
    if len(G.edges) > 0:
        edge_list = list(G.edges)
        for i, (u, v) in enumerate(edge_list):
            style = edge_styles[i] if i < len(edge_styles) else 'solid'
            color = edge_colors[i] if i < len(edge_colors) else '#555555'
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax,
                                   edge_color=color, style=style,
                                   alpha=0.5, width=1.0)

    # Узлы
    nx.draw_networkx_nodes(G, pos, ax=ax,
                       node_color=node_colors[:len(G.nodes)],
                       node_size=[float(s) for s in node_sizes[:len(G.nodes)]],
                       alpha=0.85,
                       edgecolors='white',
                       linewidths=0.5)

    # Подписи
    nx.draw_networkx_labels(G, pos, ax=ax, labels=node_labels,
                            font_size=7, font_color='white',
                            font_weight='bold')

    ax.set_facecolor('#0e1117')
    title = result.atom.name if result.atom else "КОЛЛАПС"
    ax.set_title(f'Архитектурный граф: {title}', color='white', fontsize=12, pad=15)
    ax.axis('off')

    # Легенда
    legend_patches = [
        mpatches.Patch(color='#3498db', label='Атом A (цель)'),
        mpatches.Patch(color='#2ecc71', label='Атом B (оператор)'),
        mpatches.Patch(color='#9b59b6', label='Гибридный класс'),
        mpatches.Patch(color='#e74c3c', label='Теневой класс'),
    ]
    ax.legend(handles=legend_patches, loc='upper right',
              fontsize=6, facecolor='#1a1e24', edgecolor='white',
              labelcolor='white')

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════
# AI INTERPRETER (без изменений)
# ═══════════════════════════════════════════════════════════════════

def get_ai_comment(result: SynthesisResult, api_key: str) -> str:
    """Получить комментарий от DeepSeek API."""
    if result.collapsed:
        prompt = f"""Ты — эксперт по абстрактной алгебре и теории категорий.
Синтез двух алгебраических структур привёл к КОЛЛАПСУ — все элементы носителя отождествились.
Наложено равенств: {result.equations_count}

Дай КОРОТКИЙ ответ (2-4 предложения):
1. Почему это интересно (это no-go theorem)?
2. Что именно вызвало коллапс?
3. Какие архитектурные ограничения это демонстрирует?
Отвечай на русском."""
    else:
        atom = result.atom
        prompt = f"""Ты — эксперт по абстрактной алгебре и теории категорий.
Проанализируй результат архитектурного синтеза двух структур.

Родитель A: {atom.parent_atoms[0] if atom.parent_atoms else 'нет данных'}
Родитель B: {atom.parent_atoms[1] if len(atom.parent_atoms) > 1 else 'нет данных'}
Взаимодействие: {atom.interaction}
Носитель: {', '.join(atom.carrier)} ({len(atom.carrier)} элементов)
Операции: {', '.join(f'{op}:{ar}' for op, ar in atom.operations.items())}
Классов эквивалентности: {len(result.classes)}

Дай КОРОТКИЙ ответ (2-4 предложения):
1. Что это за структура?
2. Почему она не схлопнулась?
3. Интересна ли она математически?
Отвечай на русском."""

    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 250,
                "temperature": 0.7
            },
            timeout=15
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"⚠️ Ошибка API: {response.status_code}"
    except Exception as e:
        return f"⚠️ Не удалось получить комментарий: {str(e)[:100]}"


# ═══════════════════════════════════════════════════════════════════
# REWRITING SYSTEM (без изменений)
# ═══════════════════════════════════════════════════════════════════

class RewritingSystem:
    """Система правил редукции термов."""

    def __init__(self):
        self.rules: List[Tuple[Term, Term]] = []

    def add_rule(self, left: Term, right: Term):
        """Добавить правило: left → right."""
        if len(repr(left)) >= len(repr(right)):
            self.rules.append((left, right))
        else:
            self.rules.append((right, left))

    def normalize(self, term: Term, depth: int = 0) -> Term:
        """Применить правила редукции к терму, пока возможно."""
        if depth > 100:
            return term

        if term.args:
            normalized_args = [self.normalize(arg, depth + 1) for arg in term.args]
            term = Term(term.head, normalized_args)

        for pattern, replacement in self.rules:
            mapping = self._match(pattern, term)
            if mapping is not None:
                result = replacement.substitute(mapping)
                return self.normalize(result, depth + 1)

        return term

    def _match(self, pattern: Term, term: Term) -> Optional[Dict[str, Term]]:
        """Сопоставить паттерн с термом. Вернуть подстановку или None."""
        if not pattern.args and pattern.head[0].islower():
            return {pattern.head: term}

        if pattern.head != term.head or len(pattern.args) != len(term.args):
            return None

        mapping = {}
        for p_arg, t_arg in zip(pattern.args, term.args):
            sub_match = self._match(p_arg, t_arg)
            if sub_match is None:
                return None
            for var, val in sub_match.items():
                if var in mapping:
                    if mapping[var] != val:
                        return None
                else:
                    mapping[var] = val

        return mapping

# ═══════════════════════════════════════════════════════════════════
# REDUCTION GRAPH (для теневых метрик)
# ═══════════════════════════════════════════════════════════════════

class ReductionGraph:
    """
    Строит граф редукций для терма при заданной стратегии.
    """
    
    def __init__(self, term: Term, rules: List[Tuple[Term, Term]], strategy: ReductionStrategy = ReductionStrategy.LEFT_TO_RIGHT):
        self.term = term
        self.rules = rules
        self.strategy = strategy
        self.nodes = set()
        self.edges = []
        self._build()
    
    def _build(self):
        visited = set()
        stack = [self.term]
        
        while stack:
            t = stack.pop()
            if t in visited:
                continue
            visited.add(t)
            self.nodes.add(t)
            
            for left, right in self._get_ordered_rules():
                for match in self._find_matches(t, left):
                    new_term = self._apply_match(t, match, right)
                    if new_term not in visited:
                        self.edges.append((t, new_term))
                        stack.append(new_term)
    
    def _get_ordered_rules(self):
        if self.strategy == ReductionStrategy.LEFT_TO_RIGHT:
            return self.rules
        elif self.strategy == ReductionStrategy.RIGHT_TO_LEFT:
            return list(reversed(self.rules))
        elif self.strategy == ReductionStrategy.RANDOM:
            shuffled = self.rules.copy()
            random.shuffle(shuffled)
            return shuffled
        elif self.strategy == ReductionStrategy.SHORTEST_FIRST:
            return sorted(self.rules, key=lambda r: len(repr(r[0])))
        elif self.strategy == ReductionStrategy.LONGEST_FIRST:
            return sorted(self.rules, key=lambda r: len(repr(r[0])), reverse=True)
        return self.rules
    
    def _find_matches(self, term: Term, pattern: Term) -> List[Dict[str, Term]]:
        matches = []
        self._match_recursive(term, pattern, {}, matches)
        return matches
    
    def _match_recursive(self, term: Term, pattern: Term, mapping: Dict[str, Term], matches: List):
        if not pattern.args and pattern.head[0].islower():
            if pattern.head in mapping:
                if mapping[pattern.head] == term:
                    matches.append(mapping.copy())
            else:
                mapping[pattern.head] = term
                matches.append(mapping.copy())
        elif term.args and len(term.args) == len(pattern.args):
            for i, (t_arg, p_arg) in enumerate(zip(term.args, pattern.args)):
                self._match_recursive(t_arg, p_arg, mapping, matches)
        else:
            return
    
    def _apply_match(self, term: Term, match: Dict[str, Term], pattern: Term) -> Term:
        def substitute(t: Term) -> Term:
            if not t.args and t.head[0].islower() and t.head in match:
                return match[t.head]
            return Term(t.head, [substitute(arg) for arg in t.args])
        return substitute(pattern)
    
    def get_normal_forms(self) -> List[Term]:
        outgoing = set([e[0] for e in self.edges])
        return [n for n in self.nodes if n not in outgoing]
    
    def get_graph_size(self) -> int:
        return len(self.nodes)
    
    def get_edge_count(self) -> int:
        return len(self.edges)


def generalize_rules(A: Atom) -> List[Tuple[Term, Term]]:
    """Пытается обобщить конкретные аксиомы до универсальных правил."""
    general_rules = []
    by_head = defaultdict(list)
    for left, right in A.axioms:
        if left.args:
            by_head[left.head].append((left, right))

    for head, group in by_head.items():
        if len(group) < len(A.carrier):
            continue

        for const_name, const_arity in A.operations.items():
            if const_arity != 0:
                continue
            const_term = Term(const_name, [])

            covers_all = True
            for elem in A.carrier:
                elem_term = Term(elem)
                test_term = Term(head, [const_term, elem_term])
                found = False
                for left, right in group:
                    if left == test_term and right == elem_term:
                        found = True
                        break
                if not found:
                    test_term2 = Term(head, [elem_term, const_term])
                    for left, right in group:
                        if left == test_term2 and right == elem_term:
                            found = True
                            break
                if not found:
                    covers_all = False
                    break

            if covers_all:
                var_x = Term("x")
                general_left = Term(head, [const_term, var_x])
                general_right = var_x
                general_rules.append((general_left, general_right))

    return general_rules


def add_standard_rules(rs: RewritingSystem, A: Atom, action_name: str):
    """Добавляет стандартные правила редукции для операций атома A."""
    var_x = Term("x")
    var_y = Term("y")

    for op_name, arity in A.operations.items():
        if arity != 2:
            continue

        for const_name, const_arity in A.operations.items():
            if const_arity != 0:
                continue
            const_term = Term(const_name, [])

            left_ok = all(
                any(left == Term(op_name, [const_term, Term(elem)]) and right == Term(elem)
                    for left, right in A.axioms)
                for elem in A.carrier
            )
            right_ok = all(
                any(left == Term(op_name, [Term(elem), const_term]) and right == Term(elem)
                    for left, right in A.axioms)
                for elem in A.carrier
            )

            if left_ok:
                rs.add_rule(Term(op_name, [const_term, var_x]), var_x)
            if right_ok:
                rs.add_rule(Term(op_name, [var_x, const_term]), var_x)

        for const_name, const_arity in A.operations.items():
            if const_arity != 0:
                continue
            const_term = Term(const_name, [])

            if op_name == "+" and const_name == "0":
                if not any(rule[0] == Term(op_name, [const_term, var_x]) for rule in rs.rules):
                    rs.add_rule(Term(op_name, [const_term, var_x]), var_x)
                if not any(rule[0] == Term(op_name, [var_x, const_term]) for rule in rs.rules):
                    rs.add_rule(Term(op_name, [var_x, const_term]), var_x)

            if op_name == "*" and const_name == "1":
                if not any(rule[0] == Term(op_name, [const_term, var_x]) for rule in rs.rules):
                    rs.add_rule(Term(op_name, [const_term, var_x]), var_x)
                if not any(rule[0] == Term(op_name, [var_x, const_term]) for rule in rs.rules):
                    rs.add_rule(Term(op_name, [var_x, const_term]), var_x)

# ═══════════════════════════════════════════════════════════════════
# METRICS CALCULATOR
# ═══════════════════════════════════════════════════════════════════

class MetricsCalculator:
    """Вычисляет метрики для синтезированной структуры."""
    
    @staticmethod
    def calculate(result: SynthesisResult) -> Dict[str, Any]:
        if result.collapsed:
            return {
                "collapsed": True,
                "carrier_size": 0,
                "classes_count": len(result.classes),
                "entropy": 0.0,
                "stability": 0.0,
                "collapse_steps": result.equations_count,
                "class_ratio": 0.0,
                "is_classical": False,
                "classical_type": "none"
            }
        
        atom = result.atom
        carrier_size = len(atom.carrier)
        classes_count = len(result.classes)
        equations_count = result.equations_count
        
        total_terms = sum(len(elems) for elems in result.classes.values())
        entropy = math.log(total_terms / classes_count) if classes_count > 0 else 0.0
        class_ratio = classes_count / carrier_size if carrier_size > 0 else 0.0
        stability = carrier_size / classes_count if classes_count > 0 else 0.0
        
        is_classical, classical_type = MetricsCalculator._classify_structure(atom)
        
        return {
            "collapsed": False,
            "carrier_size": carrier_size,
            "classes_count": classes_count,
            "entropy": entropy,
            "stability": stability,
            "collapse_steps": equations_count,
            "class_ratio": class_ratio,
            "is_classical": is_classical,
            "classical_type": classical_type,
            "operations": list(atom.operations.keys())
        }
    
    @staticmethod
    def _classify_structure(atom: Atom) -> Tuple[bool, str]:
        ops = set(atom.operations.keys())
        
        has_add = "+" in ops
        has_mul = "*" in ops
        has_zero = "0" in ops
        has_one = "1" in ops
        has_neg = "-" in ops
        has_inv = "inv" in ops or "⁻¹" in ops
        
        if has_add and has_zero and has_neg and not has_mul:
            return True, "group"
        if has_add and has_mul and has_zero:
            return True, "ring"
        if has_add and has_mul and has_zero and has_one and has_inv:
            return True, "field"
        if has_mul and has_one and not has_add:
            return True, "monoid"
        if "∧" in ops and "∨" in ops and "→" in ops:
            return True, "topos"
        
        return False, "hybrid"

# ═══════════════════════════════════════════════════════════════════
# SHADOW METRICS (неконфлюэнтность, стратегическая расходимость)
# ═══════════════════════════════════════════════════════════════════

class ShadowMetrics:
    @staticmethod
    def calculate(result: SynthesisResult, strategies: List[ReductionStrategy] = None) -> Dict[str, Any]:
        if strategies is None:
            strategies = list(ReductionStrategy)
        
        # Извлекаем правила из аксиом атомов
        rules = []
        if result.parent_A:
            for left, right in result.parent_A.axioms:
                rules.append((left, right))
        if result.parent_B:
            for left, right in result.parent_B.axioms:
                rules.append((left, right))
        
        all_terms = set()
        for rep, elems in result.classes.items():
            all_terms.update(elems)
        
        normal_forms = {}
        graph_sizes = {}
        
        for term in list(all_terms)[:20]:  # Ограничиваем для производительности
            for strategy in strategies:
                graph = ReductionGraph(term, rules, strategy)
                normal_forms[(term, strategy)] = len(graph.get_normal_forms())
                graph_sizes[(term, strategy)] = graph.get_graph_size()
        
        unique_factors = len(set(normal_forms.values())) if normal_forms else 0
        shadow_entropy = math.log(unique_factors) if unique_factors > 1 else 0.0
        
        # Стратегическая расходимость
        divergence = 0.0
        count = 0
        for term in list(all_terms)[:20]:
            forms = [normal_forms.get((term, s), 0) for s in strategies]
            if len(forms) > 1 and max(forms) > 0:
                divergence += max(forms) - min(forms)
                count += 1
        avg_divergence = divergence / count if count > 0 else 0.0
        
        return {
            "shadow_entropy": shadow_entropy,
            "strategy_divergence": avg_divergence,
            "stability": unique_factors,
            "unique_factors": unique_factors,
            "strategies_tested": [s.value for s in strategies]
        }

# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT LOGGER (для фазовых диаграмм)
# ═══════════════════════════════════════════════════════════════════

class ExperimentLogger:
    def __init__(self):
        self.history = []
    
    def log(self, result: SynthesisResult, context: Dict[str, Any]):
        metrics = MetricsCalculator.calculate(result)
        if not result.collapsed:
            shadow = ShadowMetrics.calculate(result)
            metrics.update(shadow)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "metrics": metrics
        }
        self.history.append(entry)
        return entry
    
    def get_phase_data(self):
        data = []
        for entry in self.history:
            m = entry["metrics"]
            data.append({
                "classes": m.get("classes_count", 0),
                "carrier": m.get("carrier_size", 0),
                "entropy": m.get("entropy", 0),
                "stability": m.get("stability", 0),
                "class_ratio": m.get("class_ratio", 0),
                "is_classical": m.get("is_classical", False),
                "type": m.get("classical_type", "none"),
                "shadow_entropy": m.get("shadow_entropy", 0),
                "divergence": m.get("strategy_divergence", 0)
            })
        return pd.DataFrame(data) if data else pd.DataFrame()
    
    def plot_phase_diagram(self):
        df = self.get_phase_data()
        if df.empty:
            fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0e1117')
            ax.text(0.5, 0.5, "Нет данных для фазовой диаграммы",
                   ha='center', va='center', color='white', fontsize=14)
            ax.set_facecolor('#0e1117')
            ax.axis('off')
            return fig
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='#0e1117')
        
        scatter = ax1.scatter(
            df["carrier"], df["classes"],
            c=df["shadow_entropy"], cmap='plasma',
            alpha=0.7, s=50
        )
        ax1.set_xlabel("Размер носителя", color='white')
        ax1.set_ylabel("Число классов", color='white')
        ax1.set_title("Классы vs Носитель (цвет = теневая энтропия)", color='white')
        ax1.set_facecolor('#1a1e24')
        ax1.tick_params(colors='white')
        plt.colorbar(scatter, ax=ax1, label='Shadow entropy')
        
        type_counts = df["type"].value_counts()
        colors = {'group': '#3498db', 'ring': '#2ecc71', 'field': '#f1c40f',
                  'monoid': '#e67e22', 'topos': '#9b59b6', 'hybrid': '#e74c3c',
                  'none': '#95a5a6'}
        pie_colors = [colors.get(t, '#95a5a6') for t in type_counts.index]
        ax2.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
                colors=pie_colors, textprops={'color': 'white', 'fontsize': 10})
        ax2.set_title("Типы структур", color='white')
        ax2.set_facecolor('#0e1117')
        
        plt.suptitle("Фазовая диаграмма структур", color='white', fontsize=16)
        plt.tight_layout()
        return fig

def build_rewriting_system(A: Atom, action_name: str) -> RewritingSystem:
    """Создать систему правил редукции из аксиом атома A."""
    rs = RewritingSystem()

    for left, right in A.axioms:
        rs.add_rule(left, right)

    general = generalize_rules(A)
    for left, right in general:
        rs.add_rule(left, right)

    add_standard_rules(rs, A, action_name)

    return rs


# ═══════════════════════════════════════════════════════════════════
# BUILT-IN LIBRARY (без изменений — весь код библиотеки)
# ═══════════════════════════════════════════════════════════════════

def create_builtin_library() -> Dict[str, Atom]:
    lib = {}

    # Z2 (additive)
    Z2 = Atom(
        name="Z2 (additive group)",
        carrier=["0", "1"],
        operations={"+": 2, "0": 0, "-": 1},
        axioms=[
            (Term("+", [Term("0"), Term("0")]), Term("0")),
            (Term("+", [Term("0"), Term("1")]), Term("1")),
            (Term("+", [Term("1"), Term("0")]), Term("1")),
            (Term("+", [Term("1"), Term("1")]), Term("0")),
            (Term("-", [Term("0")]), Term("0")),
            (Term("-", [Term("1")]), Term("1")),
        ],
        description="Циклическая группа порядка 2 (аддитивная)."
    )
    lib[Z2.name] = Z2

    # СЕМЯ — самая простая структура
    Seed = Atom(
        name="Seed (Trivial)",
        carrier=["e"],
        operations={"*": 2, "e": 0},
        axioms=[
            (Term("*", [Term("e"), Term("e")]), Term("e"))
        ],
        description="Минимальная структура: один элемент и тривиальная операция."
    )
    lib[Seed.name] = Seed

    # Z2 (ring)
    Z2_ring = Atom(
        name="Z₂ (ring)",
        carrier=["0", "1"],
        operations={"+": 2, "0": 0, "-": 1, "*": 2, "1": 0},
        axioms=[
            (Term("+", [Term("0"), Term("0")]), Term("0")),
            (Term("+", [Term("0"), Term("1")]), Term("1")),
            (Term("+", [Term("1"), Term("0")]), Term("1")),
            (Term("+", [Term("1"), Term("1")]), Term("0")),
            (Term("-", [Term("0")]), Term("0")),
            (Term("-", [Term("1")]), Term("1")),
            (Term("*", [Term("0"), Term("x")]), Term("0")),
            (Term("*", [Term("x"), Term("0")]), Term("0")),
            (Term("*", [Term("1"), Term("x")]), Term("x")),
            (Term("*", [Term("x"), Term("1")]), Term("x")),
        ],
        description="Кольцо целых чисел по модулю 2 (поле Z₂)."
    )
    lib[Z2_ring.name] = Z2_ring

    # Z3 (additive)
    Z3 = Atom(
        name="Z3 (additive group)",
        carrier=["0", "1", "2"],
        operations={"+": 2, "0": 0, "-": 1},
        axioms=[
            (Term("+", [Term("0"), Term("0")]), Term("0")),
            (Term("+", [Term("0"), Term("1")]), Term("1")),
            (Term("+", [Term("0"), Term("2")]), Term("2")),
            (Term("+", [Term("1"), Term("0")]), Term("1")),
            (Term("+", [Term("1"), Term("1")]), Term("2")),
            (Term("+", [Term("1"), Term("2")]), Term("0")),
            (Term("+", [Term("2"), Term("0")]), Term("2")),
            (Term("+", [Term("2"), Term("1")]), Term("0")),
            (Term("+", [Term("2"), Term("2")]), Term("1")),
            (Term("-", [Term("0")]), Term("0")),
            (Term("-", [Term("1")]), Term("2")),
            (Term("-", [Term("2")]), Term("1")),
        ],
        description="Циклическая группа порядка 3 (аддитивная)."
    )
    lib[Z3.name] = Z3

    # Z2 multiplicative monoid
    M2 = Atom(
        name="Z2 (multiplicative monoid)",
        carrier=["0", "1"],
        operations={"*": 2, "1": 0},
        axioms=[
            (Term("*", [Term("1"), Term("0")]), Term("0")),
            (Term("*", [Term("1"), Term("1")]), Term("1")),
            (Term("*", [Term("0"), Term("0")]), Term("0")),
            (Term("*", [Term("0"), Term("1")]), Term("0")),
        ],
        description="Мультипликативный моноид {0,1}."
    )
    lib[M2.name] = M2

    # Trivial Monoid
    TrivMon = Atom(
        name="Trivial Monoid",
        carrier=["e"],
        operations={"*": 2, "e": 0},
        axioms=[(Term("*", [Term("e"), Term("e")]), Term("e"))],
        description="Тривиальный моноид."
    )
    lib[TrivMon.name] = TrivMon

    # Boolean Lattice
    Bool = Atom(
        name="Boolean Lattice {0,1}",
        carrier=["0", "1"],
        operations={"∧": 2, "∨": 2, "¬": 1},
        axioms=[
            (Term("∧", [Term("0"), Term("0")]), Term("0")),
            (Term("∧", [Term("0"), Term("1")]), Term("0")),
            (Term("∧", [Term("1"), Term("0")]), Term("0")),
            (Term("∧", [Term("1"), Term("1")]), Term("1")),
            (Term("∨", [Term("0"), Term("0")]), Term("0")),
            (Term("∨", [Term("0"), Term("1")]), Term("1")),
            (Term("∨", [Term("1"), Term("0")]), Term("1")),
            (Term("∨", [Term("1"), Term("1")]), Term("1")),
            (Term("¬", [Term("0")]), Term("1")),
            (Term("¬", [Term("1")]), Term("0")),
        ],
        description="Булева алгебра на {0,1}."
    )
    lib[Bool.name] = Bool

    # Fair Coin
    Coin = Atom(
        name="Fair Coin (H,T)",
        carrier=["H", "T"],
        operations={},
        axioms=[],
        description="Вероятностное пространство с двумя исходами."
    )
    lib[Coin.name] = Coin

    # Free Category
    FreeCat = Atom(
        name="Free Category",
        carrier=["idA", "idB", "f", "g", "gf"],
        operations={"∘": 2},
        axioms=[
            (Term("∘", [Term("idA"), Term("f")]), Term("f")),
            (Term("∘", [Term("f"), Term("idA")]), Term("f")),
            (Term("∘", [Term("idB"), Term("g")]), Term("g")),
            (Term("∘", [Term("g"), Term("idB")]), Term("g")),
            (Term("∘", [Term("g"), Term("f")]), Term("gf")),
        ],
        description="Свободная категория на графе с 2 объектами."
    )
    lib[FreeCat.name] = FreeCat

    # Quasigroup order 3
    Q3 = Atom(
        name="Quasigroup order 3",
        carrier=["0", "1", "2"],
        operations={"*": 2},
        axioms=[
            (Term("*", [Term("0"), Term("0")]), Term("0")),
            (Term("*", [Term("0"), Term("1")]), Term("1")),
            (Term("*", [Term("0"), Term("2")]), Term("2")),
            (Term("*", [Term("1"), Term("0")]), Term("2")),
            (Term("*", [Term("1"), Term("1")]), Term("0")),
            (Term("*", [Term("1"), Term("2")]), Term("1")),
            (Term("*", [Term("2"), Term("0")]), Term("1")),
            (Term("*", [Term("2"), Term("1")]), Term("2")),
            (Term("*", [Term("2"), Term("2")]), Term("0")),
        ],
        description="Неассоциативная квазигруппа порядка 3."
    )
    lib[Q3.name] = Q3

    # Z₄ ring
    Z4_ring = Atom(
        name="Z₄ (ring)",
        carrier=["0", "1", "2", "3"],
        operations={"+": 2, "0": 0, "-": 1, "*": 2, "1": 0},
        axioms=[
            (Term("+", [Term("0"), Term("0")]), Term("0")),
            (Term("+", [Term("0"), Term("1")]), Term("1")),
            (Term("+", [Term("0"), Term("2")]), Term("2")),
            (Term("+", [Term("0"), Term("3")]), Term("3")),
            (Term("+", [Term("1"), Term("0")]), Term("1")),
            (Term("+", [Term("1"), Term("1")]), Term("2")),
            (Term("+", [Term("1"), Term("2")]), Term("3")),
            (Term("+", [Term("1"), Term("3")]), Term("0")),
            (Term("+", [Term("2"), Term("0")]), Term("2")),
            (Term("+", [Term("2"), Term("1")]), Term("3")),
            (Term("+", [Term("2"), Term("2")]), Term("0")),
            (Term("+", [Term("2"), Term("3")]), Term("1")),
            (Term("+", [Term("3"), Term("0")]), Term("3")),
            (Term("+", [Term("3"), Term("1")]), Term("0")),
            (Term("+", [Term("3"), Term("2")]), Term("1")),
            (Term("+", [Term("3"), Term("3")]), Term("2")),
            (Term("-", [Term("0")]), Term("0")),
            (Term("-", [Term("1")]), Term("3")),
            (Term("-", [Term("2")]), Term("2")),
            (Term("-", [Term("3")]), Term("1")),
            (Term("*", [Term("0"), Term("x")]), Term("0")),
            (Term("*", [Term("x"), Term("0")]), Term("0")),
            (Term("*", [Term("1"), Term("x")]), Term("x")),
            (Term("*", [Term("x"), Term("1")]), Term("x")),
            (Term("*", [Term("2"), Term("2")]), Term("0")),
            (Term("*", [Term("2"), Term("3")]), Term("2")),
            (Term("*", [Term("3"), Term("2")]), Term("2")),
            (Term("*", [Term("3"), Term("3")]), Term("1")),
        ],
        description="Кольцо вычетов по модулю 4. Не поле (2·2=0)."
    )
    lib[Z4_ring.name] = Z4_ring

    # Z₃ field
    Z3_field = Atom(
        name="Z₃ (field)",
        carrier=["0", "1", "2"],
        operations={"+": 2, "0": 0, "-": 1, "*": 2, "1": 0},
        axioms=[
            (Term("+", [Term("0"), Term("0")]), Term("0")),
            (Term("+", [Term("0"), Term("1")]), Term("1")),
            (Term("+", [Term("0"), Term("2")]), Term("2")),
            (Term("+", [Term("1"), Term("0")]), Term("1")),
            (Term("+", [Term("1"), Term("1")]), Term("2")),
            (Term("+", [Term("1"), Term("2")]), Term("0")),
            (Term("+", [Term("2"), Term("0")]), Term("2")),
            (Term("+", [Term("2"), Term("1")]), Term("0")),
            (Term("+", [Term("2"), Term("2")]), Term("1")),
            (Term("-", [Term("0")]), Term("0")),
            (Term("-", [Term("1")]), Term("2")),
            (Term("-", [Term("2")]), Term("1")),
            (Term("*", [Term("0"), Term("x")]), Term("0")),
            (Term("*", [Term("x"), Term("0")]), Term("0")),
            (Term("*", [Term("1"), Term("x")]), Term("x")),
            (Term("*", [Term("x"), Term("1")]), Term("x")),
            (Term("*", [Term("2"), Term("2")]), Term("1")),
        ],
        description="Поле из трёх элементов (Z/3Z)."
    )
    lib[Z3_field.name] = Z3_field

    # V₂ over GF(2)
    V2_GF2 = Atom(
        name="V₂ over GF(2)",
        carrier=["0", "e1", "e2", "e1+e2"],
        operations={"+": 2, "0": 0, "-": 1},
        axioms=[
            (Term("+", [Term("0"), Term("0")]), Term("0")),
            (Term("+", [Term("0"), Term("e1")]), Term("e1")),
            (Term("+", [Term("0"), Term("e2")]), Term("e2")),
            (Term("+", [Term("0"), Term("e1+e2")]), Term("e1+e2")),
            (Term("+", [Term("e1"), Term("0")]), Term("e1")),
            (Term("+", [Term("e1"), Term("e1")]), Term("0")),
            (Term("+", [Term("e1"), Term("e2")]), Term("e1+e2")),
            (Term("+", [Term("e1"), Term("e1+e2")]), Term("e2")),
            (Term("+", [Term("e2"), Term("0")]), Term("e2")),
            (Term("+", [Term("e2"), Term("e1")]), Term("e1+e2")),
            (Term("+", [Term("e2"), Term("e2")]), Term("0")),
            (Term("+", [Term("e2"), Term("e1+e2")]), Term("e1")),
            (Term("+", [Term("e1+e2"), Term("0")]), Term("e1+e2")),
            (Term("+", [Term("e1+e2"), Term("e1")]), Term("e2")),
            (Term("+", [Term("e1+e2"), Term("e2")]), Term("e1")),
            (Term("+", [Term("e1+e2"), Term("e1+e2")]), Term("0")),
            (Term("-", [Term("x")]), Term("x")),
        ],
        description="Двумерное векторное пространство над GF(2)."
    )
    lib[V2_GF2.name] = V2_GF2

    # Dual numbers over Z₂
    Dual_Z2 = Atom(
        name="Dual numbers over Z₂",
        carrier=["0", "1", "ε", "1+ε"],
        operations={"+": 2, "0": 0, "-": 1, "*": 2, "1": 0},
        axioms=[
            (Term("+", [Term("0"), Term("0")]), Term("0")),
            (Term("+", [Term("0"), Term("1")]), Term("1")),
            (Term("+", [Term("0"), Term("ε")]), Term("ε")),
            (Term("+", [Term("0"), Term("1+ε")]), Term("1+ε")),
            (Term("+", [Term("1"), Term("0")]), Term("1")),
            (Term("+", [Term("1"), Term("1")]), Term("0")),
            (Term("+", [Term("1"), Term("ε")]), Term("1+ε")),
            (Term("+", [Term("1"), Term("1+ε")]), Term("ε")),
            (Term("+", [Term("ε"), Term("0")]), Term("ε")),
            (Term("+", [Term("ε"), Term("1")]), Term("1+ε")),
            (Term("+", [Term("ε"), Term("ε")]), Term("0")),
            (Term("+", [Term("ε"), Term("1+ε")]), Term("1")),
            (Term("+", [Term("1+ε"), Term("0")]), Term("1+ε")),
            (Term("+", [Term("1+ε"), Term("1")]), Term("ε")),
            (Term("+", [Term("1+ε"), Term("ε")]), Term("1")),
            (Term("+", [Term("1+ε"), Term("1+ε")]), Term("0")),
            (Term("-", [Term("x")]), Term("x")),
            (Term("*", [Term("0"), Term("x")]), Term("0")),
            (Term("*", [Term("x"), Term("0")]), Term("0")),
            (Term("*", [Term("1"), Term("x")]), Term("x")),
            (Term("*", [Term("x"), Term("1")]), Term("x")),
            (Term("*", [Term("ε"), Term("ε")]), Term("0")),
            (Term("*", [Term("ε"), Term("1+ε")]), Term("ε")),
            (Term("*", [Term("1+ε"), Term("ε")]), Term("ε")),
            (Term("*", [Term("1+ε"), Term("1+ε")]), Term("1")),
        ],
        description="Кольцо дуальных чисел над Z₂. ε²=0. Содержит нильпотент."
    )
    lib[Dual_Z2.name] = Dual_Z2

    # Heyting Algebra (3-chain)
    Heyting3 = Atom(
        name="Heyting Algebra (3-chain)",
        carrier=["0", "a", "1"],
        operations={"∧": 2, "∨": 2, "→": 2, "0": 0, "1": 0},
        axioms=[
            (Term("∧", [Term("0"), Term("x")]), Term("0")),
            (Term("∧", [Term("x"), Term("0")]), Term("0")),
            (Term("∧", [Term("1"), Term("x")]), Term("x")),
            (Term("∧", [Term("x"), Term("1")]), Term("x")),
            (Term("∧", [Term("a"), Term("a")]), Term("a")),
            (Term("∨", [Term("0"), Term("x")]), Term("x")),
            (Term("∨", [Term("x"), Term("0")]), Term("x")),
            (Term("∨", [Term("1"), Term("x")]), Term("1")),
            (Term("∨", [Term("x"), Term("1")]), Term("1")),
            (Term("∨", [Term("a"), Term("a")]), Term("a")),
            (Term("→", [Term("0"), Term("x")]), Term("1")),
            (Term("→", [Term("1"), Term("x")]), Term("x")),
            (Term("→", [Term("x"), Term("1")]), Term("1")),
            (Term("→", [Term("a"), Term("0")]), Term("0")),
            (Term("→", [Term("a"), Term("a")]), Term("1")),
            (Term("→", [Term("a"), Term("1")]), Term("1")),
            (Term("→", [Term("0"), Term("a")]), Term("1")),
            (Term("→", [Term("1"), Term("a")]), Term("a")),
            (Term("→", [Term("1"), Term("0")]), Term("0")),
        ],
        description="Трёхэлементная алгебра Гейтинга (интуиционистская логика)."
    )
    lib[Heyting3.name] = Heyting3

    # MV-algebra (3-valued Łukasiewicz)
    MV3 = Atom(
        name="MV-algebra (3-valued Łukasiewicz)",
        carrier=["0", "½", "1"],
        operations={"⊕": 2, "¬": 1, "0": 0},
        axioms=[
            (Term("⊕", [Term("0"), Term("0")]), Term("0")),
            (Term("⊕", [Term("0"), Term("½")]), Term("½")),
            (Term("⊕", [Term("0"), Term("1")]), Term("1")),
            (Term("⊕", [Term("½"), Term("0")]), Term("½")),
            (Term("⊕", [Term("½"), Term("½")]), Term("1")),
            (Term("⊕", [Term("½"), Term("1")]), Term("1")),
            (Term("⊕", [Term("1"), Term("0")]), Term("1")),
            (Term("⊕", [Term("1"), Term("½")]), Term("1")),
            (Term("⊕", [Term("1"), Term("1")]), Term("1")),
            (Term("¬", [Term("0")]), Term("1")),
            (Term("¬", [Term("½")]), Term("½")),
            (Term("¬", [Term("1")]), Term("0")),
        ],
        description="Трёхзначная MV-алгебра Łukasiewicza. Допустима небулева логика."
    )
    lib[MV3.name] = MV3

    # Free Category (3-object chain)
    Cat3 = Atom(
        name="Free Category (3-object chain)",
        carrier=["id_A", "id_B", "id_C", "f", "g", "g∘f"],
        operations={"∘": 2},
        axioms=[
            (Term("∘", [Term("id_A"), Term("f")]), Term("f")),
            (Term("∘", [Term("f"), Term("id_A")]), Term("f")),
            (Term("∘", [Term("id_B"), Term("f")]), Term("f")),
            (Term("∘", [Term("f"), Term("id_B")]), Term("f")),
            (Term("∘", [Term("id_B"), Term("g")]), Term("g")),
            (Term("∘", [Term("g"), Term("id_B")]), Term("g")),
            (Term("∘", [Term("id_C"), Term("g")]), Term("g")),
            (Term("∘", [Term("g"), Term("id_C")]), Term("g")),
            (Term("∘", [Term("g"), Term("f")]), Term("g∘f")),
            (Term("∘", [Term("id_B"), Term("g∘f")]), Term("g∘f")),
            (Term("∘", [Term("g∘f"), Term("id_A")]), Term("g∘f")),
        ],
        description="Свободная категория на графе A→B→C (3 объекта)."
    )
    lib[Cat3.name] = Cat3

    # Category with Zero Object
    CatZero = Atom(
        name="Category with Zero Object",
        carrier=["id_Z", "id_A", "z_AZ", "z_ZA", "z_A^2"],
        operations={"∘": 2},
        axioms=[
            (Term("∘", [Term("id_Z"), Term("z_AZ")]), Term("z_AZ")),
            (Term("∘", [Term("z_AZ"), Term("id_A")]), Term("z_AZ")),
            (Term("∘", [Term("id_A"), Term("z_ZA")]), Term("z_ZA")),
            (Term("∘", [Term("z_ZA"), Term("id_Z")]), Term("z_ZA")),
            (Term("∘", [Term("z_ZA"), Term("z_AZ")]), Term("z_A^2")),
            (Term("∘", [Term("id_Z"), Term("z_A^2")]), Term("z_A^2")),
        ],
        description="Категория с нулевым объектом Z."
    )
    lib[CatZero.name] = CatZero

    # Topos Sh({p,q})
    Topos2 = Atom(
        name="Topos Sh({p,q})",
        carrier=["∅_∅", "∅_{q}", "{p}_∅", "{p}_{q}"],
        operations={"∧": 2, "∨": 2, "→": 2, "Ω": 0},
        axioms=[
            (Term("∧", [Term("∅_∅"), Term("x")]), Term("∅_∅")),
            (Term("∧", [Term("x"), Term("∅_∅")]), Term("∅_∅")),
            (Term("∧", [Term("{p}_{q}"), Term("x")]), Term("x")),
            (Term("∧", [Term("x"), Term("{p}_{q}")]), Term("x")),
            (Term("∨", [Term("∅_∅"), Term("x")]), Term("x")),
            (Term("∨", [Term("x"), Term("∅_∅")]), Term("x")),
            (Term("∨", [Term("{p}_{q}"), Term("x")]), Term("{p}_{q}")),
            (Term("∨", [Term("x"), Term("{p}_{q}")]), Term("{p}_{q}")),
        ],
        description="Топос пучков на двухточечном дискретном пространстве (Set²)."
    )
    lib[Topos2.name] = Topos2

    # sl(2, Z₂)
    sl2_Z2 = Atom(
        name="sl(2, Z₂)",
        carrier=["0", "e", "f", "h", "e+f", "e+h", "f+h", "e+f+h"],
        operations={"+": 2, "0": 0, "-": 1, "[_,_]": 2},
        axioms=[
            (Term("+", [Term("0"), Term("x")]), Term("x")),
            (Term("+", [Term("x"), Term("0")]), Term("x")),
            (Term("+", [Term("e"), Term("e")]), Term("0")),
            (Term("+", [Term("f"), Term("f")]), Term("0")),
            (Term("+", [Term("h"), Term("h")]), Term("0")),
            (Term("+", [Term("e"), Term("f")]), Term("e+f")),
            (Term("+", [Term("e"), Term("h")]), Term("e+h")),
            (Term("+", [Term("f"), Term("h")]), Term("f+h")),
            (Term("+", [Term("e"), Term("e+f")]), Term("f")),
            (Term("+", [Term("e"), Term("e+h")]), Term("h")),
            (Term("+", [Term("e"), Term("f+h")]), Term("e+f+h")),
            (Term("+", [Term("f"), Term("e+f")]), Term("e")),
            (Term("+", [Term("f"), Term("f+h")]), Term("h")),
            (Term("+", [Term("f"), Term("e+h")]), Term("e+f+h")),
            (Term("+", [Term("h"), Term("e+f")]), Term("e+f+h")),
            (Term("+", [Term("h"), Term("e+h")]), Term("e")),
            (Term("+", [Term("h"), Term("f+h")]), Term("f")),
            (Term("+", [Term("e+f"), Term("e+h")]), Term("f+h")),
            (Term("+", [Term("e+f"), Term("f+h")]), Term("e+h")),
            (Term("+", [Term("e+h"), Term("f+h")]), Term("e+f")),
            (Term("-", [Term("x")]), Term("x")),
            (Term("[_,_]", [Term("e"), Term("f")]), Term("h")),
            (Term("[_,_]", [Term("f"), Term("e")]), Term("h")),
            (Term("[_,_]", [Term("h"), Term("e")]), Term("e")),
            (Term("[_,_]", [Term("e"), Term("h")]), Term("e")),
            (Term("[_,_]", [Term("h"), Term("f")]), Term("f")),
            (Term("[_,_]", [Term("f"), Term("h")]), Term("f")),
            (Term("[_,_]", [Term("e"), Term("e")]), Term("0")),
            (Term("[_,_]", [Term("f"), Term("f")]), Term("0")),
            (Term("[_,_]", [Term("h"), Term("h")]), Term("0")),
            (Term("[_,_]", [Term("0"), Term("x")]), Term("0")),
            (Term("[_,_]", [Term("x"), Term("0")]), Term("0")),
        ],
        description="Алгебра Ли sl(2) над полем Z₂. 8 элементов."
    )
    lib[sl2_Z2.name] = sl2_Z2

    # Jordan algebra
    Jordan_Z2 = Atom(
        name="Jordan algebra (sym 2×2/Z₂)",
        carrier=["0", "I", "A", "B", "I+A", "I+B", "A+B", "I+A+B"],
        operations={"+": 2, "0": 0, "-": 1, "◦": 2},
        axioms=[
            (Term("+", [Term("0"), Term("x")]), Term("x")),
            (Term("+", [Term("x"), Term("0")]), Term("x")),
            (Term("+", [Term("I"), Term("I")]), Term("0")),
            (Term("+", [Term("A"), Term("A")]), Term("0")),
            (Term("+", [Term("B"), Term("B")]), Term("0")),
            (Term("+", [Term("I"), Term("A")]), Term("I+A")),
            (Term("+", [Term("I"), Term("B")]), Term("I+B")),
            (Term("+", [Term("A"), Term("B")]), Term("A+B")),
            (Term("+", [Term("I"), Term("A+B")]), Term("I+A+B")),
            (Term("+", [Term("A"), Term("I+B")]), Term("I+A+B")),
            (Term("+", [Term("B"), Term("I+A")]), Term("I+A+B")),
            (Term("+", [Term("I+A"), Term("I+B")]), Term("A+B")),
            (Term("+", [Term("I+A"), Term("A+B")]), Term("I+B")),
            (Term("+", [Term("I+B"), Term("A+B")]), Term("I+A")),
            (Term("-", [Term("x")]), Term("x")),
            (Term("◦", [Term("0"), Term("x")]), Term("0")),
            (Term("◦", [Term("x"), Term("0")]), Term("0")),
            (Term("◦", [Term("I"), Term("x")]), Term("x")),
            (Term("◦", [Term("x"), Term("I")]), Term("x")),
            (Term("◦", [Term("A"), Term("A")]), Term("A")),
            (Term("◦", [Term("B"), Term("B")]), Term("I")),
            (Term("◦", [Term("A"), Term("B")]), Term("B")),
            (Term("◦", [Term("B"), Term("A")]), Term("B")),
            (Term("◦", [Term("A"), Term("I+A")]), Term("A")),
            (Term("◦", [Term("B"), Term("I+B")]), Term("I+B")),
            (Term("◦", [Term("◦", [Term("A"), Term("B")]), Term("B")]), Term("I")),
            (Term("◦", [Term("A"), Term("◦", [Term("B"), Term("B")])]), Term("A")),
        ],
        description="Йорданова алгебра симметрических 2×2 матриц над Z₂."
    )
    lib[Jordan_Z2.name] = Jordan_Z2

    # Cayley-Dickson over Z₂
    Cayley_Z2 = Atom(
        name="Cayley-Dickson over Z₂",
        carrier=["0", "1", "e1", "e2", "e1e2", "1+e1", "1+e2", "1+e1e2", "e1+e2", "e1+e1e2", "e2+e1e2", "1+e1+e2", "1+e1+e1e2", "1+e2+e1e2", "e1+e2+e1e2", "1+e1+e2+e1e2"],
        operations={"+": 2, "0": 0, "-": 1, "*": 2},
        axioms=[
            (Term("+", [Term("0"), Term("x")]), Term("x")),
            (Term("+", [Term("x"), Term("0")]), Term("x")),
            (Term("-", [Term("x")]), Term("x")),
            (Term("*", [Term("0"), Term("x")]), Term("0")),
            (Term("*", [Term("x"), Term("0")]), Term("0")),
            (Term("*", [Term("1"), Term("x")]), Term("x")),
            (Term("*", [Term("x"), Term("1")]), Term("x")),
            (Term("*", [Term("e1"), Term("e1")]), Term("1")),
            (Term("*", [Term("e2"), Term("e2")]), Term("1")),
            (Term("*", [Term("e1e2"), Term("e1e2")]), Term("1")),
            (Term("*", [Term("e1"), Term("e2")]), Term("e1e2")),
            (Term("*", [Term("e2"), Term("e1")]), Term("e1e2")),
            (Term("*", [Term("e1"), Term("e1e2")]), Term("e2")),
            (Term("*", [Term("e1e2"), Term("e1")]), Term("e2")),
            (Term("*", [Term("e2"), Term("e1e2")]), Term("e1")),
            (Term("*", [Term("e1e2"), Term("e2")]), Term("e1")),
        ],
        description="Алгебра Кэли-Диксона (кватернионы) над Z₂."
    )
    lib[Cayley_Z2.name] = Cayley_Z2

    # Idempotent algebra Z₂[x]/(x²+x)
    Idempotent_Z2 = Atom(
        name="Idempotent algebra Z₂[x]/(x²+x)",
        carrier=["0", "1", "x", "1+x"],
        operations={"+": 2, "0": 0, "-": 1, "*": 2, "1": 0},
        axioms=[
            (Term("+", [Term("0"), Term("a")]), Term("a")),
            (Term("+", [Term("a"), Term("0")]), Term("a")),
            (Term("+", [Term("1"), Term("1")]), Term("0")),
            (Term("+", [Term("x"), Term("x")]), Term("0")),
            (Term("+", [Term("1+x"), Term("1+x")]), Term("0")),
            (Term("+", [Term("1"), Term("x")]), Term("1+x")),
            (Term("+", [Term("1"), Term("1+x")]), Term("x")),
            (Term("+", [Term("x"), Term("1+x")]), Term("1")),
            (Term("-", [Term("a")]), Term("a")),
            (Term("*", [Term("0"), Term("a")]), Term("0")),
            (Term("*", [Term("a"), Term("0")]), Term("0")),
            (Term("*", [Term("1"), Term("a")]), Term("a")),
            (Term("*", [Term("a"), Term("1")]), Term("a")),
            (Term("*", [Term("x"), Term("x")]), Term("x")),
            (Term("*", [Term("x"), Term("1+x")]), Term("0")),
            (Term("*", [Term("1+x"), Term("x")]), Term("0")),
            (Term("*", [Term("1+x"), Term("1+x")]), Term("1+x")),
        ],
        description="Алгебра с идемпотентом x²=x над Z₂. Изоморфна Z₂⊕Z₂."
    )
    lib[Idempotent_Z2.name] = Idempotent_Z2

    # Topos Set^{Z₂}
    Topos_Z2 = Atom(
        name="Topos Set^{Z₂}",
        carrier=["0_0", "0_1", "1_0", "1_1"],
        operations={"∧": 2, "∨": 2, "→": 2, "¬": 1, "Ω": 0},
        axioms=[
            (Term("∧", [Term("0_0"), Term("x")]), Term("0_0")),
            (Term("∧", [Term("0_1"), Term("x")]), Term("0_1")),
            (Term("∧", [Term("1_0"), Term("x")]), Term("x")),
            (Term("∧", [Term("1_1"), Term("x")]), Term("x")),
            (Term("∨", [Term("0_0"), Term("x")]), Term("x")),
            (Term("∨", [Term("0_1"), Term("x")]), Term("x")),
            (Term("∨", [Term("1_0"), Term("x")]), Term("1_0")),
            (Term("∨", [Term("1_1"), Term("x")]), Term("1_1")),
            (Term("¬", [Term("0_0")]), Term("0_0")),
            (Term("¬", [Term("0_1")]), Term("1_0")),
            (Term("¬", [Term("1_0")]), Term("0_1")),
            (Term("¬", [Term("1_1")]), Term("1_1")),
        ],
        description="Топос функторов Set^{Z₂} — простейший нетривиальный топос с действием группы."
    )
    lib[Topos_Z2.name] = Topos_Z2

    # Sheaf on 3-point chain
    Sheaf3 = Atom(
        name="Sheaf on 3-point chain",
        carrier=["∅", "{a}", "{a,b}", "{a,b,c}"],
        operations={"∧": 2, "∨": 2, "→": 2, "1": 0},
        axioms=[
            (Term("∧", [Term("∅"), Term("x")]), Term("∅")),
            (Term("∧", [Term("{a,b,c}"), Term("x")]), Term("x")),
            (Term("∧", [Term("{a}"), Term("{a,b}")]), Term("{a}")),
            (Term("∨", [Term("∅"), Term("x")]), Term("x")),
            (Term("∨", [Term("{a,b,c}"), Term("x")]), Term("{a,b,c}")),
            (Term("∨", [Term("{a}"), Term("{a,b}")]), Term("{a,b}")),
            (Term("→", [Term("∅"), Term("x")]), Term("{a,b,c}")),
            (Term("→", [Term("x"), Term("{a,b,c}")]), Term("{a,b,c}")),
        ],
        description="Пучок на трёхточечном пространстве с топологией цепи."
    )
    lib[Sheaf3.name] = Sheaf3

    # Markov Category (2 states)
    Markov2 = Atom(
        name="Markov Category (2 states)",
        carrier=["s0", "s1", "p_id", "p_flip", "p_0", "p_1"],
        operations={"∘": 2, "⊗": 2, "id": 0},
        axioms=[
            (Term("∘", [Term("id"), Term("x")]), Term("x")),
            (Term("∘", [Term("x"), Term("id")]), Term("x")),
            (Term("∘", [Term("p_id"), Term("p_id")]), Term("p_id")),
            (Term("∘", [Term("p_id"), Term("p_flip")]), Term("p_flip")),
            (Term("∘", [Term("p_flip"), Term("p_id")]), Term("p_flip")),
            (Term("∘", [Term("p_flip"), Term("p_flip")]), Term("p_id")),
            (Term("∘", [Term("p_0"), Term("x")]), Term("p_0")),
            (Term("∘", [Term("p_1"), Term("x")]), Term("p_1")),
        ],
        description="Марковская категория на двух состояниях с вероятностными переходами."
    )
    lib[Markov2.name] = Markov2

    # Effect Algebra (3 elements)
    Effect3 = Atom(
        name="Effect Algebra (3 elements)",
        carrier=["0", "½", "1"],
        operations={"⊕": 2, "¬": 1, "0": 0, "1": 0},
        axioms=[
            (Term("⊕", [Term("0"), Term("0")]), Term("0")),
            (Term("⊕", [Term("0"), Term("½")]), Term("½")),
            (Term("⊕", [Term("½"), Term("0")]), Term("½")),
            (Term("⊕", [Term("0"), Term("1")]), Term("1")),
            (Term("⊕", [Term("1"), Term("0")]), Term("1")),
            (Term("⊕", [Term("½"), Term("½")]), Term("1")),
            (Term("¬", [Term("0")]), Term("1")),
            (Term("¬", [Term("½")]), Term("½")),
            (Term("¬", [Term("1")]), Term("0")),
        ],
        description="Трёхэлементная эффект-алгебра. Модель квантовой логики."
    )
    lib[Effect3.name] = Effect3

    # Subobject Classifier Ω (2 values)
    Omega2 = Atom(
        name="Subobject Classifier Ω (2 values)",
        carrier=["false", "true"],
        operations={"∧": 2, "∨": 2, "⇒": 2, "true": 0, "false": 0},
        axioms=[
            (Term("∧", [Term("false"), Term("x")]), Term("false")),
            (Term("∧", [Term("x"), Term("false")]), Term("false")),
            (Term("∧", [Term("true"), Term("x")]), Term("x")),
            (Term("∧", [Term("x"), Term("true")]), Term("x")),
            (Term("∨", [Term("false"), Term("x")]), Term("x")),
            (Term("∨", [Term("x"), Term("false")]), Term("x")),
            (Term("∨", [Term("true"), Term("x")]), Term("true")),
            (Term("∨", [Term("x"), Term("true")]), Term("true")),
            (Term("⇒", [Term("false"), Term("x")]), Term("true")),
            (Term("⇒", [Term("true"), Term("false")]), Term("false")),
            (Term("⇒", [Term("true"), Term("true")]), Term("true")),
            (Term("⇒", [Term("x"), Term("true")]), Term("true")),
            (Term("⇒", [Term("x"), Term("x")]), Term("true")),
        ],
        description="Классификатор подобъектов Ω топоса Set."
    )
    lib[Omega2.name] = Omega2

    # Атом "Two" — минимальная структура с двумя элементами
    Two = Atom(
        name="Two Elements (0, e)",
        carrier=["0", "e"],
        operations={
            "*": 2,   # бинарная операция
            "0": 0,   # нейтральный элемент (пока номинально)
            "e": 0    # потенциальный единичный элемент
        },
        axioms=[
            # 1. Тривиальные умножения (чтобы они не схлопнулись в 0)
            (Term("*", [Term("0"), Term("0")]), Term("0")),
            (Term("*", [Term("0"), Term("e")]), Term("e")),
            (Term("*", [Term("e"), Term("0")]), Term("e")),
            (Term("*", [Term("e"), Term("e")]), Term("e")),
        
            # 2. Константы как элементы
            (Term("0", []), Term("0")),
            (Term("e", []), Term("e")),
        ],
        description="Минимальная структура с двумя элементами 0 и e, где операция * не схлопывает их."
    )
    lib[Two.name] = Two

    # Free Magma (1 generator, depth ≤ 2)
    Magma1 = Atom(
        name="Free Magma (1 generator, depth ≤ 2)",
        carrier=["x", "x*x", "(x*x)*x", "x*(x*x)", "(x*x)*(x*x)"],
        operations={"*": 2},
        axioms=[
            (Term("*", [Term("x"), Term("x")]), Term("x*x")),
            (Term("*", [Term("x*x"), Term("x")]), Term("(x*x)*x")),
            (Term("*", [Term("x"), Term("x*x")]), Term("x*(x*x)")),
            (Term("*", [Term("x*x"), Term("x*x")]), Term("(x*x)*(x*x)")),
        ],
        description="Свободная магма с одним генератором. Глубина деревьев ≤ 2."
    )
    lib[Magma1.name] = Magma1

    # Magma order 3 (partial assoc)
    Magma3 = Atom(
        name="Magma order 3 (partial assoc)",
        carrier=["a", "b", "c"],
        operations={"*": 2},
        axioms=[
            (Term("*", [Term("a"), Term("a")]), Term("a")),
            (Term("*", [Term("a"), Term("b")]), Term("b")),
            (Term("*", [Term("a"), Term("c")]), Term("c")),
            (Term("*", [Term("b"), Term("a")]), Term("b")),
            (Term("*", [Term("b"), Term("b")]), Term("c")),
            (Term("*", [Term("b"), Term("c")]), Term("a")),
            (Term("*", [Term("c"), Term("a")]), Term("c")),
            (Term("*", [Term("c"), Term("b")]), Term("a")),
            (Term("*", [Term("c"), Term("c")]), Term("b")),
        ],
        description="Магма порядка 3. Частично ассоциативна."
    )
    lib[Magma3.name] = Magma3

    # Commutative Magma order 3
    CommMagma = Atom(
        name="Commutative Magma order 3",
        carrier=["0", "1", "2"],
        operations={"·": 2},
        axioms=[
            (Term("·", [Term("0"), Term("0")]), Term("1")),
            (Term("·", [Term("0"), Term("1")]), Term("2")),
            (Term("·", [Term("1"), Term("0")]), Term("2")),
            (Term("·", [Term("0"), Term("2")]), Term("0")),
            (Term("·", [Term("2"), Term("0")]), Term("0")),
            (Term("·", [Term("1"), Term("1")]), Term("0")),
            (Term("·", [Term("1"), Term("2")]), Term("1")),
            (Term("·", [Term("2"), Term("1")]), Term("1")),
            (Term("·", [Term("2"), Term("2")]), Term("2")),
        ],
        description="Коммутативная магма порядка 3. Коммутативна, но НЕ ассоциативна."
    )
    lib[CommMagma.name] = CommMagma

    # Z ⊕ Z (free abelian, truncated)
    Z2_free_ab = Atom(
        name="Z ⊕ Z (free abelian, truncated)",
        carrier=["0", "a", "b", "a+b", "-a", "-b", "-a+b", "a-b", "-a-b"],
        operations={"+": 2, "0": 0, "-": 1},
        axioms=[
            (Term("+", [Term("0"), Term("x")]), Term("x")),
            (Term("+", [Term("x"), Term("0")]), Term("x")),
            (Term("+", [Term("a"), Term("b")]), Term("a+b")),
            (Term("+", [Term("b"), Term("a")]), Term("a+b")),
            (Term("+", [Term("a"), Term("-a")]), Term("0")),
            (Term("+", [Term("b"), Term("-b")]), Term("0")),
            (Term("+", [Term("a+b"), Term("-a")]), Term("b")),
            (Term("+", [Term("a+b"), Term("-b")]), Term("a")),
            (Term("+", [Term("-a"), Term("-b")]), Term("-a-b")),
            (Term("-", [Term("0")]), Term("0")),
            (Term("-", [Term("a")]), Term("-a")),
            (Term("-", [Term("b")]), Term("-b")),
            (Term("-", [Term("a+b")]), Term("-a-b")),
            (Term("-", [Term("-a")]), Term("a")),
            (Term("-", [Term("-b")]), Term("b")),
        ],
        description="Свободная абелева группа ранга 2 (усечённая)."
    )
    lib[Z2_free_ab.name] = Z2_free_ab

    # Diagonal C*-algebra over Z₂
    Cstar2 = Atom(
        name="Diagonal C*-algebra over Z₂",
        carrier=["0", "I", "E1", "E2", "I+E1", "I+E2", "E1+E2", "I+E1+E2"],
        operations={"+": 2, "0": 0, "-": 1, "*": 2, "†": 1},
        axioms=[
            (Term("+", [Term("0"), Term("x")]), Term("x")),
            (Term("+", [Term("x"), Term("0")]), Term("x")),
            (Term("-", [Term("x")]), Term("x")),
            (Term("+", [Term("I"), Term("E1")]), Term("I+E1")),
            (Term("+", [Term("I"), Term("E2")]), Term("I+E2")),
            (Term("+", [Term("E1"), Term("E2")]), Term("E1+E2")),
            (Term("*", [Term("0"), Term("x")]), Term("0")),
            (Term("*", [Term("x"), Term("0")]), Term("0")),
            (Term("*", [Term("I"), Term("x")]), Term("x")),
            (Term("*", [Term("x"), Term("I")]), Term("x")),
            (Term("*", [Term("E1"), Term("E1")]), Term("E1")),
            (Term("*", [Term("E2"), Term("E2")]), Term("E2")),
            (Term("*", [Term("E1"), Term("E2")]), Term("0")),
            (Term("*", [Term("E2"), Term("E1")]), Term("0")),
            (Term("†", [Term("0")]), Term("0")),
            (Term("†", [Term("I")]), Term("I")),
            (Term("†", [Term("E1")]), Term("E1")),
            (Term("†", [Term("E2")]), Term("E2")),
        ],
        description="Коммутативная C*-алгебра диагональных 2×2 матриц над Z₂."
    )
    lib[Cstar2.name] = Cstar2

    # GF(4)
    GF4 = Atom(
        name="GF(4) — Finite Field of 4 elements",
        carrier=["0", "1", "ω", "ω+1"],
        operations={"+": 2, "0": 0, "-": 1, "*": 2, "1": 0},
        axioms=[
            (Term("+", [Term("0"), Term("x")]), Term("x")),
            (Term("+", [Term("1"), Term("1")]), Term("0")),
            (Term("+", [Term("ω"), Term("ω")]), Term("0")),
            (Term("+", [Term("ω+1"), Term("ω+1")]), Term("0")),
            (Term("+", [Term("1"), Term("ω")]), Term("ω+1")),
            (Term("+", [Term("1"), Term("ω+1")]), Term("ω")),
            (Term("+", [Term("ω"), Term("ω+1")]), Term("1")),
            (Term("-", [Term("x")]), Term("x")),
            (Term("*", [Term("0"), Term("x")]), Term("0")),
            (Term("*", [Term("x"), Term("0")]), Term("0")),
            (Term("*", [Term("1"), Term("x")]), Term("x")),
            (Term("*", [Term("x"), Term("1")]), Term("x")),
            (Term("*", [Term("ω"), Term("ω")]), Term("ω+1")),
            (Term("*", [Term("ω"), Term("ω+1")]), Term("1")),
            (Term("*", [Term("ω+1"), Term("ω")]), Term("1")),
            (Term("*", [Term("ω+1"), Term("ω+1")]), Term("ω")),
        ],
        description="Конечное поле из 4 элементов. ω² = ω+1."
    )
    lib[GF4.name] = GF4

    # Quaternions over Z₃ (basis only)
    Quat_Z3 = Atom(
        name="Quaternions over Z₃ (basis only)",
        carrier=["0", "1", "-1", "i", "-i", "j", "-j", "k", "-k"],
        operations={"+": 2, "0": 0, "-": 1, "*": 2},
        axioms=[
            (Term("+", [Term("0"), Term("x")]), Term("x")),
            (Term("+", [Term("1"), Term("-1")]), Term("0")),
            (Term("+", [Term("i"), Term("-i")]), Term("0")),
            (Term("+", [Term("j"), Term("-j")]), Term("0")),
            (Term("+", [Term("k"), Term("-k")]), Term("0")),
            (Term("-", [Term("0")]), Term("0")),
            (Term("-", [Term("1")]), Term("-1")),
            (Term("-", [Term("-1")]), Term("1")),
            (Term("-", [Term("i")]), Term("-i")),
            (Term("-", [Term("j")]), Term("-j")),
            (Term("-", [Term("k")]), Term("-k")),
            (Term("*", [Term("0"), Term("x")]), Term("0")),
            (Term("*", [Term("1"), Term("x")]), Term("x")),
            (Term("*", [Term("i"), Term("i")]), Term("-1")),
            (Term("*", [Term("j"), Term("j")]), Term("-1")),
            (Term("*", [Term("k"), Term("k")]), Term("-1")),
            (Term("*", [Term("i"), Term("j")]), Term("k")),
            (Term("*", [Term("j"), Term("i")]), Term("-k")),
            (Term("*", [Term("j"), Term("k")]), Term("i")),
            (Term("*", [Term("k"), Term("j")]), Term("-i")),
            (Term("*", [Term("k"), Term("i")]), Term("j")),
            (Term("*", [Term("i"), Term("k")]), Term("-j")),
        ],
        description="Алгебра кватернионов над Z₃ (некоммутативная алгебра с делением)."
    )
    lib[Quat_Z3.name] = Quat_Z3

    # GF(2)^2 — двумерное векторное пространство над полем из двух элементов
    V2_GF2 = Atom(
        name="V₂ over GF(2) for Minkowski",
        carrier=["0", "e1", "e2", "e1+e2"],  # 4 вектора
        operations={
            "+": 2,   # сложение векторов
            "0": 0,   # нулевой вектор
            "-": 1    # обратный элемент (в GF(2) совпадает с самим собой)
        },
        axioms=[
            # Сложение с нулём
            (Term("+", [Term("0"), Term("0")]), Term("0")),
            (Term("+", [Term("0"), Term("e1")]), Term("e1")),
            (Term("+", [Term("0"), Term("e2")]), Term("e2")),
            (Term("+", [Term("0"), Term("e1+e2")]), Term("e1+e2")),
        
            # Коммутативность + сам себе обратный
            (Term("+", [Term("e1"), Term("e1")]), Term("0")),
            (Term("+", [Term("e2"), Term("e2")]), Term("0")),
            (Term("+", [Term("e1+e2"), Term("e1+e2")]), Term("0")),
        
            # Сложение базисных векторов
            (Term("+", [Term("e1"), Term("e2")]), Term("e1+e2")),
            (Term("+", [Term("e2"), Term("e1")]), Term("e1+e2")),
        
            # Сложение с e1+e2
            (Term("+", [Term("e1"), Term("e1+e2")]), Term("e2")),
            (Term("+", [Term("e2"), Term("e1+e2")]), Term("e1")),
            (Term("+", [Term("e1+e2"), Term("e1")]), Term("e2")),
            (Term("+", [Term("e1+e2"), Term("e2")]), Term("e1")),
        
            # Обратный элемент (в GF(2) обратный = сам себе)
            (Term("-", [Term("0")]), Term("0")),
            (Term("-", [Term("e1")]), Term("e1")),
            (Term("-", [Term("e2")]), Term("e2")),
            (Term("-", [Term("e1+e2")]), Term("e1+e2")),
        ],
        description="Двумерное векторное пространство над GF(2) — конечная модель для пространства Минковского."
    )
    lib[V2_GF2.name] = V2_GF2

    # Группа вращений SO(2) над GF(2) — группа порядка 2
    SO2_GF2 = Atom(
        name="SO(2) over GF(2)",
        carrier=["I", "R"],  # I — тождество, R — отражение/вращение на 180°
        operations={
            "∘": 2,   # композиция преобразований
            "I": 0,   # нейтральный элемент (тождество)
            "inv": 1  # обратное преобразование
        },
        axioms=[
            # Тождество — нейтральный элемент
            (Term("∘", [Term("I"), Term("I")]), Term("I")),
            (Term("∘", [Term("I"), Term("R")]), Term("R")),
            (Term("∘", [Term("R"), Term("I")]), Term("R")),
        
            # R ∘ R = I (вращение на 180° дважды даёт тождество)
            (Term("∘", [Term("R"), Term("R")]), Term("I")),
        
            # Обратные элементы
            (Term("inv", [Term("I")]), Term("I")),
            (Term("inv", [Term("R")]), Term("R")),
        ],
        description="Конечная группа вращений SO(2) над GF(2) — модель для группы Лоренца."
    )
    lib[SO2_GF2.name] = SO2_GF2

    # V₄ for Minkowski (4-мерное пространство-время над GF(2))
    V4_Minkowski = Atom(
        name="V₄ for Minkowski (t, x, y, z)",
        carrier=["0", "t", "x", "y", "z", "t+x", "t+y", "t+z", "x+y", "x+z", "y+z", "t+x+y", "t+x+z", "t+y+z", "x+y+z", "t+x+y+z"],
        operations={
            "+": 2,   # сложение векторов
            "0": 0,   # нулевой вектор
            "-": 1    # обратный элемент (в GF(2) совпадает с самим собой)
        },
        axioms=[
            # Сложение с нулём
            (Term("+", [Term("0"), Term("0")]), Term("0")),
            (Term("+", [Term("0"), Term("t")]), Term("t")),
            (Term("+", [Term("0"), Term("x")]), Term("x")),
            (Term("+", [Term("0"), Term("y")]), Term("y")),
            (Term("+", [Term("0"), Term("z")]), Term("z")),
            (Term("+", [Term("0"), Term("t+x")]), Term("t+x")),
            (Term("+", [Term("0"), Term("t+y")]), Term("t+y")),
            (Term("+", [Term("0"), Term("t+z")]), Term("t+z")),
            (Term("+", [Term("0"), Term("x+y")]), Term("x+y")),
            (Term("+", [Term("0"), Term("x+z")]), Term("x+z")),
            (Term("+", [Term("0"), Term("y+z")]), Term("y+z")),
            (Term("+", [Term("0"), Term("t+x+y")]), Term("t+x+y")),
            (Term("+", [Term("0"), Term("t+x+z")]), Term("t+x+z")),
            (Term("+", [Term("0"), Term("t+y+z")]), Term("t+y+z")),
            (Term("+", [Term("0"), Term("x+y+z")]), Term("x+y+z")),
            (Term("+", [Term("0"), Term("t+x+y+z")]), Term("t+x+y+z")),
        
            # Сам себе обратный (GF(2))
            (Term("+", [Term("t"), Term("t")]), Term("0")),
            (Term("+", [Term("x"), Term("x")]), Term("0")),
            (Term("+", [Term("y"), Term("y")]), Term("0")),
            (Term("+", [Term("z"), Term("z")]), Term("0")),
        
            # Сложение базисных векторов
            (Term("+", [Term("t"), Term("x")]), Term("t+x")),
            (Term("+", [Term("t"), Term("y")]), Term("t+y")),
            (Term("+", [Term("t"), Term("z")]), Term("t+z")),
            (Term("+", [Term("x"), Term("y")]), Term("x+y")),
            (Term("+", [Term("x"), Term("z")]), Term("x+z")),
            (Term("+", [Term("y"), Term("z")]), Term("y+z")),
        
            # Остальные комбинации (программа выведет их автоматически через замкнутость)
            # Но для безопасности можно добавить все остальные 256 комбинаций
            # Это слишком долго писать вручную, поэтому мы полагаемся на 
            # систему переписывания, которая выведет их из коммутативности и ассоциативности.
        
            # Обратный элемент (в GF(2) равен самому себе)
            (Term("-", [Term("0")]), Term("0")),
            (Term("-", [Term("t")]), Term("t")),
            (Term("-", [Term("x")]), Term("x")),
            (Term("-", [Term("y")]), Term("y")),
            (Term("-", [Term("z")]), Term("z")),
            (Term("-", [Term("t+x")]), Term("t+x")),
            (Term("-", [Term("t+y")]), Term("t+y")),
            (Term("-", [Term("t+z")]), Term("t+z")),
            (Term("-", [Term("x+y")]), Term("x+y")),
            (Term("-", [Term("x+z")]), Term("x+z")),
            (Term("-", [Term("y+z")]), Term("y+z")),
            (Term("-", [Term("t+x+y")]), Term("t+x+y")),
            (Term("-", [Term("t+x+z")]), Term("t+x+z")),
            (Term("-", [Term("t+y+z")]), Term("t+y+z")),
            (Term("-", [Term("x+y+z")]), Term("x+y+z")),
            (Term("-", [Term("t+x+y+z")]), Term("t+x+y+z")),
        ],
        description="4-мерное векторное пространство над GF(2) для моделирования пространства Минковского с сигнатурой."
    )
    lib[V4_Minkowski.name] = V4_Minkowski

    # Группа Лоренца O(1,1) над GF(2) — 4 элемента
    Lorentz_GF2 = Atom(
        name="Lorentz group (1+1) over GF(2)",
        carrier=["I", "B", "P", "BP"],  # I — тождество, B — буст, P — пространственное отражение, BP — их композиция
        operations={
            "∘": 2,   # композиция
            "I": 0,   # нейтральный элемент
            "inv": 1  # обратный элемент
        },
        axioms=[
            # I — нейтральный элемент
            (Term("∘", [Term("I"), Term("I")]), Term("I")),
            (Term("∘", [Term("I"), Term("B")]), Term("B")),
            (Term("∘", [Term("I"), Term("P")]), Term("P")),
            (Term("∘", [Term("I"), Term("BP")]), Term("BP")),
            (Term("∘", [Term("B"), Term("I")]), Term("B")),
            (Term("∘", [Term("P"), Term("I")]), Term("P")),
            (Term("∘", [Term("BP"), Term("I")]), Term("BP")),
        
            # B — буст (меняет время и пространство)
            (Term("∘", [Term("B"), Term("B")]), Term("I")),   # B^2 = I (в GF(2) буст дважды даёт тождество)
            (Term("∘", [Term("B"), Term("P")]), Term("BP")),  # B ∘ P = BP
            (Term("∘", [Term("P"), Term("B")]), Term("BP")),  # P ∘ B = BP
         
            # P — пространственное отражение
            (Term("∘", [Term("P"), Term("P")]), Term("I")),   # P^2 = I
        
            # BP — композиция (B ∘ P)
            (Term("∘", [Term("BP"), Term("BP")]), Term("I")), # (BP)^2 = I
            (Term("∘", [Term("BP"), Term("B")]), Term("P")),  # BP ∘ B = P
            (Term("∘", [Term("BP"), Term("P")]), Term("B")),  # BP ∘ P = B
            (Term("∘", [Term("B"), Term("BP")]), Term("P")),  # B ∘ BP = P
            (Term("∘", [Term("P"), Term("BP")]), Term("B")),  # P ∘ BP = B
        
            # Обратные элементы
            (Term("inv", [Term("I")]), Term("I")),
            (Term("inv", [Term("B")]), Term("B")),
            (Term("inv", [Term("P")]), Term("P")),
            (Term("inv", [Term("BP")]), Term("BP")),
        ],
        description="Группа Лоренца O(1,1) над GF(2) — модель для преобразований пространства Минковского."
    )
    lib[Lorentz_GF2.name] = Lorentz_GF2
    # Дикое пространство Минковского — 16 элементов, с нарушенной аддитивностью
    Wild_Minkowski = Atom(
        name="Wild Minkowski (t, x, y, z)",
        carrier=["0", "t", "x", "y", "z", "t+x", "t+y", "t+z", "x+y", "x+z", "y+z", "t+x+y", "t+x+z", "t+y+z", "x+y+z", "t+x+y+z"],
        operations={
            "+": 2,   # сложение с нарушенной аддитивностью
            "0": 0,   # нейтральный элемент
            "-": 1    # обратный элемент (в GF(2) равен самому себе)
        },
        axioms=[
            # 1. Нейтральный элемент работает
            (Term("+", [Term("0"), Term("0")]), Term("0")),
            (Term("+", [Term("0"), Term("t")]), Term("t")),
            (Term("+", [Term("0"), Term("x")]), Term("x")),
            (Term("+", [Term("0"), Term("y")]), Term("y")),
            (Term("+", [Term("0"), Term("z")]), Term("z")),
            (Term("+", [Term("t"), Term("0")]), Term("t")),
            (Term("+", [Term("x"), Term("0")]), Term("x")),
            (Term("+", [Term("y"), Term("0")]), Term("y")),
            (Term("+", [Term("z"), Term("0")]), Term("z")),

            # 2. Сам себе обратный (GF(2))
            (Term("+", [Term("t"), Term("t")]), Term("0")),
            (Term("+", [Term("x"), Term("x")]), Term("0")),
            (Term("+", [Term("y"), Term("y")]), Term("0")),
            (Term("+", [Term("z"), Term("z")]), Term("0")),

            # 3. Базовые комбинации (часть из них мы НЕ задаём — они уйдут в тень)
            (Term("+", [Term("t"), Term("x")]), Term("t+x")),
            (Term("+", [Term("t"), Term("y")]), Term("t+y")),
            (Term("+", [Term("t"), Term("z")]), Term("t+z")),
            (Term("+", [Term("x"), Term("y")]), Term("x+y")),
            (Term("+", [Term("x"), Term("z")]), Term("x+z")),
            (Term("+", [Term("y"), Term("z")]), Term("y+z")),

            # 4. Обратный элемент (в GF(2) равен самому себе)
            (Term("-", [Term("0")]), Term("0")),
            (Term("-", [Term("t")]), Term("t")),
            (Term("-", [Term("x")]), Term("x")),
            (Term("-", [Term("y")]), Term("y")),
            (Term("-", [Term("z")]), Term("z")),
            (Term("-", [Term("t+x")]), Term("t+x")),
            (Term("-", [Term("t+y")]), Term("t+y")),
            (Term("-", [Term("t+z")]), Term("t+z")),
            (Term("-", [Term("x+y")]), Term("x+y")),
            (Term("-", [Term("x+z")]), Term("x+z")),
            (Term("-", [Term("y+z")]), Term("y+z")),
            (Term("-", [Term("t+x+y")]), Term("t+x+y")),
            (Term("-", [Term("t+x+z")]), Term("t+x+z")),
            (Term("-", [Term("t+y+z")]), Term("t+y+z")),
            (Term("-", [Term("x+y+z")]), Term("x+y+z")),
            (Term("-", [Term("t+x+y+z")]), Term("t+x+y+z")),
        ],
        description="Дикое пространство Минковского с нарушенной аддитивностью."
    )
    lib[Wild_Minkowski.name] = Wild_Minkowski

    # Дикая группа Лоренца — с действием, которое заведомо уходит в тень
    Wild_Lorentz = Atom(
        name="Wild Lorentz Group (B, P)",
        carrier=["I", "B", "P", "BP"],
        operations={
            "∘": 2,
            "I": 0,
            "inv": 1
        },
        axioms=[
            # Тождество
            (Term("∘", [Term("I"), Term("I")]), Term("I")),
            (Term("∘", [Term("I"), Term("B")]), Term("B")),
            (Term("∘", [Term("I"), Term("P")]), Term("P")),
            (Term("∘", [Term("I"), Term("BP")]), Term("BP")),
            (Term("∘", [Term("B"), Term("I")]), Term("B")),
            (Term("∘", [Term("P"), Term("I")]), Term("P")),
            (Term("∘", [Term("BP"), Term("I")]), Term("BP")),

            # Буст
            (Term("∘", [Term("B"), Term("B")]), Term("I")),
            (Term("∘", [Term("B"), Term("P")]), Term("BP")),
            (Term("∘", [Term("P"), Term("B")]), Term("BP")),
 
            # Отражение
            (Term("∘", [Term("P"), Term("P")]), Term("I")),

            # Композиция
            (Term("∘", [Term("BP"), Term("BP")]), Term("I")),
            (Term("∘", [Term("BP"), Term("B")]), Term("P")),
            (Term("∘", [Term("BP"), Term("P")]), Term("B")),
            (Term("∘", [Term("B"), Term("BP")]), Term("P")),
            (Term("∘", [Term("P"), Term("BP")]), Term("B")),

            # Обратные
            (Term("inv", [Term("I")]), Term("I")),
            (Term("inv", [Term("B")]), Term("B")),
            (Term("inv", [Term("P")]), Term("P")),
            (Term("inv", [Term("BP")]), Term("BP")),
        ],
        description="Дикая группа Лоренца — действие уходит в тень."
    )
    lib[Wild_Lorentz.name] = Wild_Lorentz


    print(f"✅ Total structures in library: {len(lib)}")

    return lib


# ═══════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Hybrid Synthesis Lab", page_icon="🧬", layout="wide")
st.title("🧬 Hybrid Synthesis Laboratory")
st.markdown("**Лаборатория архитектурного синтеза алгебраических структур**")

if 'library' not in st.session_state:
    st.session_state.library = create_builtin_library()

if 'synthesis_history' not in st.session_state:
    st.session_state.synthesis_history = []

# Инициализация контекста (чтобы не терялся при rerun)
if 'identifications' not in st.session_state:
    st.session_state.identifications = []
if 'custom_eqs' not in st.session_state:
    st.session_state.custom_eqs = []

# Боковая панель
with st.sidebar:
    st.header("📚 Библиотека")
    lib = st.session_state.library
    names = sorted(lib.keys())
    st.caption(f"Структур: {len(names)}")

    atom_a_name = st.selectbox("Атом A (цель)", names, key="atom_a")
    atom_b_name = st.selectbox("Атом B (оператор)", names, key="atom_b")

    # Сброс контекста при смене атомов (чтобы не было мусора)
    if st.session_state.get("_last_a") != atom_a_name or st.session_state.get("_last_b") != atom_b_name:
        st.session_state.identifications = []
        st.session_state.custom_eqs = []
        st.session_state["_last_a"] = atom_a_name
        st.session_state["_last_b"] = atom_b_name

    # ── Выбор действия ──────────────────────────────────────
    st.subheader("⚡ Действие")
    action_presets = {
        "· (умножение)": "·",
        "* (умножение)": "*",
        "+ (сложение)": "+",
        "∘ (композиция)": "∘",
        "act (действие группы)": "act",
        "scalar_mul (скалярное умножение)": "scalar_mul",
        "evolve (эволюция)": "evolve",
        "measure (измерение)": "measure",
        "random_choice (случайный выбор)": "random_choice",
        "apply (применение)": "apply",
        "symmetry (симметрия)": "symmetry",
        "⇒ (импликация)": "⇒",
        "⊕ (сильная дизъюнкция)": "⊕",
        "[_,_] (скобка Ли)": "[_,_]",
        "◦ (йорданово произведение)": "◦",
        "⊗ (тензорное произведение)": "⊗",
        "conj (сопряжение)": "conj",
        "trivial (тривиальное действие)": "trivial",
        "▸ своё действие": "custom",
    }

    preset_label = st.selectbox(
        "Тип действия",
        list(action_presets.keys()),
        help="Выберите тип взаимодействия. «▸ своё действие» — ввести вручную."
    )

    if action_presets[preset_label] == "custom":
        action_name = st.text_input("Название своего действия", "·", key="custom_action")
    else:
        action_name = action_presets[preset_label]
        st.caption(f"Текущее действие: `{action_name}`")

    # ═══════════════════════════════════════════════════════════════
    # ПРОСТОЙ КОНТЕКСТ — одно большое окно для равенств (как ты просил)
    # ═══════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🔗 Контекст (равенства)")
    st.caption(
        "Вставь сюда все нужные равенства сразу (по одному на строку). "
        "Формат: `левая = правая`. Можно копировать/вставлять целыми блоками."
    )

    # Получаем элементы
    A = lib.get(atom_a_name)
    B = lib.get(atom_b_name)
    elem_A = A.carrier if A else []
    elem_B = B.carrier if B else []

    # === 1. ОТОЖДЕСТВЛЕНИЯ ЭЛЕМЕНТОВ (оставил минимально) ===
    with st.expander("Отождествления элементов (A ≡ B) — опционально"):
        col_add1, col_add2, col_add3 = st.columns([2.2, 2.2, 0.8])
        with col_add1:
            a_elem = st.selectbox("Из A", elem_A, key="a_elem_new")
        with col_add2:
            b_elem = st.selectbox("Из B", elem_B, key="b_elem_new")
        with col_add3:
            if st.button("➕ Добавить", key="add_ident"):
                pair = (a_elem, b_elem)
                if pair not in st.session_state.identifications:
                    st.session_state.identifications.append(pair)
                    st.rerun()

        if st.session_state.identifications:
            for i, (ae, be) in enumerate(st.session_state.identifications[:]):
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(f"`{ae}` ≡ `{be}`")
                with c2:
                    if st.button("🗑️", key=f"del_id_{i}"):
                        st.session_state.identifications.pop(i)
                        st.rerun()

    # === 2. ГЛАВНОЕ: ОДНО БОЛЬШОЕ ОКНО ДЛЯ ВСЕХ РАВЕНСТВ ===
    default_text = "\n".join([f"{l} = {r}" for l, r in st.session_state.custom_eqs]) if st.session_state.custom_eqs else ""
    context_text = st.text_area(
        "Равенства (копируй/вставляй сюда)",
        value=default_text,
        height=220,
        placeholder="""*(1, 0) = 0
*(1, 1) = 1
*(0, 0) = 0
*(0, 1) = 0
0 = 0
1 = 1""",
        help="Каждое равенство на отдельной строке. При синтезе всё автоматически разберётся."
    )

    aggressive_norm = st.checkbox(
        "Агрессивная нормализация (полностью упрощать по твоим равенствам)",
        value=False,
        help="По умолчанию выключено — чтобы видеть все артефакты и 'дикие' результаты. Включи только когда хочешь чистый классический итог."
    )

        # 🔮 Анализ теней (неконфлюэнтность, стратегии)
    shadow_norm = st.checkbox(
        "🔮 Анализ теней (неконфлюэнтность, стратегии)",
        value=False,
        help="Вычисляет теневую энтропию и стратегическую расходимость на графе редукций."
    )

    # Кнопка очистки (опционально)
    if st.button("🧹 Очистить равенства", key="clear_eqs"):
        st.session_state.custom_eqs = []
        st.rerun()

    # Кнопка очистки
    if st.button("🧹 Очистить весь контекст", type="secondary"):
        st.session_state.identifications = []
        st.session_state.custom_eqs = []
        st.rerun()

    # API-ключ
    st.markdown("---")
    st.subheader("🤖 AI-интерпретатор")
    api_key = st.text_input("DeepSeek API ключ", type="password",
                            help="Получить на platform.deepseek.com")

    if st.button("🚀 Синтезировать", type="primary", use_container_width=True):
        # === ПАРСИНГ ТЕКСТОВОГО ПОЛЯ В РАВЕНСТВА ===
        parsed_custom = []
        for line in context_text.strip().split("\n"):
            line = line.strip()
            if not line or "=" not in line:
                continue
            left, right = line.split("=", 1)
            left = left.strip()
            right = right.strip()
            if left and right and left != right:
                parsed_custom.append((left, right))

        # Сохраняем в session_state (чтобы показывалось в textarea)
        st.session_state.custom_eqs = parsed_custom

        A_obj = lib[atom_a_name]
        B_obj = lib[atom_b_name]
        with st.spinner("Синтез..."):
            result = synthesize(
                A_obj, B_obj, action_name,
                user_equations = st.session_state.identifications if st.session_state.identifications else None,
                custom_equations = parsed_custom if parsed_custom else None,
                aggressive_normalization = aggressive_norm   
               
            )
            

        

        st.session_state.last_result = result

        if result.collapsed:
            st.error("💥 Коллапс!")
        else:
            st.success(f"✅ {result.atom.name}")
            lib[result.atom.name] = result.atom
            st.rerun()
            
   
# Основная область
tab1, tab2 = st.tabs(["🔬 Результат", "📖 Библиотека"])

with tab1:
    if 'last_result' not in st.session_state:
        st.info("Выполните синтез в боковой панели.")
    else:
        result = st.session_state.last_result
        A = lib[atom_a_name] if 'atom_a_name' in locals() else None
        B = lib[atom_b_name] if 'atom_b_name' in locals() else None

        if result.collapsed:
            st.error("💥 АРХИТЕКТУРА КОЛЛАПСИРОВАЛА")
            st.markdown(
                "**Все элементы носителя отождествлены.** "
                "Данная конфигурация структур и взаимодействия математически невозможна — это **no-go theorem**."
            )
            st.metric("Количество наложенных равенств", result.equations_count)

            with st.expander("🧾 Вынужденные равенства (все нетривиальные)", expanded=False):
                st.caption("Эти равенства были наложены коуравнителем (показаны все классы с >1 элементом):")
                st.write(f"Всего равенств: {result.equations_count}")
        else:
            atom = result.atom
            st.success(f"✅ **{atom.name}** — структура успешно синтезирована")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Элементов в носителе", len(atom.carrier))
            with col2:
                st.metric("Операций", len(atom.operations))
            with col3:
                st.metric("Равенств", result.equations_count)
            with col4:
                st.metric("Классов эквивал.", len(result.classes))

            st.markdown(f"**Родители:** {', '.join(atom.parent_atoms)}")
            st.markdown(f"**Взаимодействие:** {atom.interaction}")

            # ── ВКЛАДКИ ДЛЯ ДЕТАЛЕЙ ─────────────────────────────
            detail_tab1, detail_tab2 = st.tabs(["📊 Таблицы и граф", "🧾 Классы эквивалентности"])

            with detail_tab1:
                st.subheader("🧬 Носитель новой структуры")
                st.markdown(
                    "Каждый элемент — это **класс эквивалентности**, "
                    "возникший при склейке:"
                )
                for i, elem in enumerate(atom.carrier):
                    st.write(f"**{i+1}.** `{elem}`")

                st.subheader("🧮 Сохранённые операции")
                ops_list = [
                    f"`{op}` (арность {ar})" for op, ar in atom.operations.items()
                ]
                st.markdown(", ".join(ops_list))

                if action_name in atom.operations and B is not None:
                    st.subheader(f"📊 Таблица действия `{action_name}` (B × A → A)")
                    st.markdown(
                        "Показывает результат **b · a** для каждого b из оператора "
                        "и каждого a из целевой структуры:"
                    )
                    table_data = []
                    rs = build_rewriting_system(A, action_name)
                    for b_elem in B.carrier:
                        row = [f"**{b_elem}**"]
                        for a_elem in atom.carrier:
                            action_term = Term(action_name, [Term(b_elem), Term(a_elem)])
                            norm_action = rs.normalize(action_term)

                            # Приоритет — поиск через CongruenceClosure
                            found_rep = None
                            if result.cc and norm_action in result.cc.parent:
                                root = result.cc.find(norm_action)
                                for rep, elems in result.classes.items():
                                    if rep == root:
                                        found_rep = rep
                                        break
                                    if any(result.cc.find(e) == root for e in elems):
                                        found_rep = rep
                                        break

                            # Запасная проверка
                            if found_rep is None:
                                for rep, elems in result.classes.items():
                                    if norm_action == rep or norm_action in elems:
                                        found_rep = rep
                                        break
                                    if repr(norm_action) == repr(rep) or any(repr(norm_action) == repr(e) for e in elems):
                                        found_rep = rep
                                        break

                            if found_rep is not None:
                                row.append(f"`{repr(found_rep)}`")
                            else:
                                row.append("—")
                        table_data.append(row)

                    for row in table_data:
                        st.write(" | ".join(row))
                    st.caption("Прочерк означает, что терм не попал ни в один класс (редкий случай).")

                for op_name, arity in atom.operations.items():
                    if op_name == action_name:
                        continue
                    if arity == 2:
                        st.subheader(f"📊 Таблица Кэли для `{op_name}`")
                        table_data = []
                        for a1 in atom.carrier:
                            row = [f"**{a1}**"]
                            for a2 in atom.carrier:
                                term = Term(op_name, [Term(a1), Term(a2)])
                                found = False
                                if result.cc and term in result.cc.parent:
                                    root = result.cc.find(term)
                                    norm_root = rs.normalize(root)
                                    for rep, elems in result.classes.items():
                                        if rep == norm_root or norm_root in elems or any(
                                            result.cc.find(e) == result.cc.find(norm_root) for e in elems
                                        ):
                                            row.append(f"`{repr(rep)}`")
                                            found = True
                                            break
                                    if not found:
                                        for rep, elems in result.classes.items():
                                            if rep == root or root in elems or any(
                                                result.cc.find(e) == result.cc.find(root) for e in elems
                                            ):
                                                row.append(f"`{repr(rep)}`")
                                                found = True
                                                break
                                if not found:
                                    rs = build_rewriting_system(A, action_name)
                                    norm_term = rs.normalize(term)
                                    for rep, elems in result.classes.items():
                                        if norm_term in elems or any(
                                            repr(e) == repr(norm_term) for e in elems
                                        ):
                                            row.append(f"`{repr(rep)}`")
                                            found = True
                                            break
                                if not found:
                                    row.append("—")
                            table_data.append(row)
                        for row in table_data:
                            st.write(" | ".join(row))

                # ── ГРАФ ──────────────────────────────────────
                st.subheader("🔗 Архитектурный граф структуры")
                st.caption(
                    "Узлы — классы эквивалентности. Размер — количество термов в классе. "
                    "Синий = атом A, Зелёный = атом B, Фиолетовый = гибрид, Красный = теневой класс. "
                    "Сплошные линии — замкнутые операции, пунктир — теневые, оранжевый пунктир — действие."
                )
                try:
                    fig = build_synthesis_graph(result)
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.warning(f"Не удалось построить граф: {e}")

            with detail_tab2:
                # ── Вынужденные равенства ──
                with st.expander("🧾 Вынужденные равенства (все нетривиальные)", expanded=False):
                    st.caption(
                        "Эти равенства были наложены коуравнителем. "
                        "Показаны все классы с более чем одним элементом."
                    )
                    equality_text = f"Всего равенств: {result.equations_count}\n"
                    nontrivial = {
                        rep: elems
                        for rep, elems in result.classes.items()
                        if len(elems) > 1
                    }
                    if nontrivial:
                        equality_text += "Нетривиальные отождествления:\n"
                        for rep, elems in list(nontrivial.items()):
                            equality_text += f"{repr(rep)} ← {', '.join(map(repr, elems[:8]))}"
                            if len(elems) > 8:
                                equality_text += " ..."
                            equality_text += "\n"
                    else:
                        equality_text += "Все классы тривиальны.\n"

                    st.code(equality_text, language="text")

                # ── Полные классы эквивалентности ──
                with st.expander("🔬 Полные классы эквивалентности", expanded=False):
                    st.caption(
                        "Каждый класс — это множество термов, отождествлённых коуравнителем."
                    )
                    classes_text = ""
                    for rep, elems in sorted(
                        result.classes.items(), key=lambda x: repr(x[0])
                    ):
                        elems_str = ", ".join(map(repr, elems[:20]))
                        if len(elems) > 20:
                            elems_str += f" ... (+{len(elems) - 20})"
                        classes_text += f"{repr(rep)} → {{{elems_str}}}\n"

                    st.code(classes_text, language="text")

            st.subheader("🔍 Проверка алгебраических свойств")
            if "+" in atom.operations and action_name in atom.operations:
                st.markdown(
                    "✅ **Дистрибутивность** обеспечивается коуравнителем "
                    "(встроена в классы эквивалентности)"
                )
            if any(
                op in atom.operations
                for op in ["0", "1", "e"]
            ):
                st.markdown("✅ **Нейтральный элемент** присутствует")
            if len(atom.operations) == len(A.operations):
                st.markdown("✅ Все операции целевой структуры **сохранены**")
            elif len(atom.operations) > len(A.operations):
                st.markdown("✅ Операции целевой структуры сохранены + добавлено действие")
            else:
                st.markdown(
                    "⚠️ Часть операций целевой структуры **исчезла** при синтезе "
                    "(возможно, они несовместимы с действием)"
                )

        st.markdown("---")
        if api_key:
            if st.button("🤖 Интерпретировать результат (AI)", type="secondary"):
                with st.spinner("DeepSeek анализирует синтез..."):
                    if result.collapsed:
                        prompt = f"""Ты — эксперт по абстрактной алгебре и теории категорий.
Синтез двух алгебраических структур привёл к КОЛЛАПСУ — все элементы носителя отождествились.
Наложено равенств: {result.equations_count}

Дай КОРОТКИЙ ответ (2-4 предложения):
1. Почему это интересно (это no-go theorem)?
2. Что именно вызвало коллапс?
3. Какие архитектурные ограничения это демонстрирует?
Отвечай на русском."""
                    else:
                        atom = result.atom
                        carrier_str = ", ".join(atom.carrier[:10])
                        ops_str = ", ".join(
                            f"{op}:{ar}" for op, ar in atom.operations.items()
                        )
                        nontrivial_count = sum(
                            1
                            for elems in result.classes.values()
                            if len(elems) > 1
                        )
                        prompt = f"""Ты — эксперт по абстрактной алгебре и теории категорий.
Проанализируй результат архитектурного синтеза двух структур.

Родитель A: {atom.parent_atoms[0] if atom.parent_atoms else 'неизвестно'}
Родитель B: {atom.parent_atoms[1] if len(atom.parent_atoms) > 1 else 'неизвестно'}
Взаимодействие: {atom.interaction}
Носитель: {carrier_str} ({len(atom.carrier)} элементов)
Операции: {ops_str}
Нетривиальных классов эквивалентности: {nontrivial_count}
Всего классов: {len(result.classes)}

Дай КОРОТКИЙ ответ (2-4 предложения):
1. Что это за структура?
2. Почему она не схлопнулась?
3. Интересна ли она математически?
Отвечай на русском."""
                    comment = get_ai_comment(result, api_key)
                    st.info(f"💬 **Комментарий AI:**\n\n{comment}")
        else:
            st.caption(
                "Введите API-ключ DeepSeek в боковой панели для AI-интерпретации."
            )

with tab2:
    for name in sorted(lib.keys()):
        atom = lib[name]
        with st.expander(f"{'🔷' if atom.is_synthetic else '💠'} {name}"):
            st.write(f"Носитель: {', '.join(atom.carrier)}")
            st.write(f"Операции: {', '.join(f'{op}:{ar}' for op, ar in atom.operations.items())}")

st.markdown("---")
st.caption("Hybrid Synthesis Laboratory v2.1 | L. Shcherbakov (2026)")
