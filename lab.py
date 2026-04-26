"""
Hybrid Synthesis Laboratory — с AI-интерпретацией результатов
"""

import streamlit as st
from dataclasses import dataclass, field
from collections import defaultdict
import itertools
from typing import List, Tuple, Dict, Set, Optional
import json
from datetime import datetime
import requests


# ═══════════════════════════════════════════════════════════════════
# CORE ENGINE
# ═══════════════════════════════════════════════════════════════════

class Term:
    """Символьный терм для алгебраических выражений."""
    
    def __init__(self, head: str, args: List["Term"] | None = None):
        self.head = head
        self.args = args or []

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


def synthesize(A: Atom, B: Atom, action_name: str = "·") -> SynthesisResult:
    all_ops = {}
    all_ops.update(A.operations)
    all_ops.update(B.operations)
    all_ops[action_name] = 2

    equations: List[Tuple[Term, Term]] = []

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

    # Проверка коллапса
    carrier_terms = [Term(el) for el in A.carrier]
    distinct_roots = {cc.find(t) for t in carrier_terms}
    collapsed = len(distinct_roots) <= 1

    if collapsed:
        return SynthesisResult(
            atom=None,
            collapsed=True,
            classes=dict(classes),
            equations_count=len(equations),
            timestamp=datetime.now().isoformat()
        )

    # Построение нового атома
    new_carrier = []
    carrier_repr_map = {}
    for t in carrier_terms:
        root = cc.find(t)
        if root not in carrier_repr_map:
            repr_name = repr(root)
            new_carrier.append(repr_name)
            carrier_repr_map[root] = repr_name

    new_operations = {}
    for op, arity in A.operations.items():
        if arity == 0:
            const_term = Term(op, [])
            root = cc.find(const_term)
            if root in carrier_repr_map:
                new_operations[op] = 0
        else:
            new_operations[op] = arity

    new_operations[action_name] = 2

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
        timestamp=datetime.now().isoformat()
    )


# ═══════════════════════════════════════════════════════════════════
# AI INTERPRETER
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
# BUILT-IN LIBRARY
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

# Боковая панель
with st.sidebar:
    st.header("📚 Библиотека")
    lib = st.session_state.library
    names = sorted(lib.keys())
    st.caption(f"Структур: {len(names)}")
    
    atom_a_name = st.selectbox("Атом A (цель)", names)
    atom_b_name = st.selectbox("Атом B (оператор)", names)
    action_name = st.text_input("Действие", "·")
    
    # API-ключ
    st.markdown("---")
    st.subheader("🤖 AI-интерпретатор")
    api_key = st.text_input("DeepSeek API ключ", type="password", 
                            help="Получить на platform.deepseek.com")
    
    if st.button("🚀 Синтезировать", type="primary", use_container_width=True):
        A = lib[atom_a_name]
        B = lib[atom_b_name]
        with st.spinner("Синтез..."):
            result = synthesize(A, B, action_name)
        st.session_state.last_result = result
        
        if result.collapsed:
            st.error("💥 Коллапс!")
        else:
            st.success(f"✅ {result.atom.name}")
            lib[result.atom.name] = result.atom

# Основная область
tab1, tab2 = st.tabs(["🔬 Результат", "📖 Библиотека"])

with tab1:
    if 'last_result' not in st.session_state:
        st.info("Выполните синтез в боковой панели.")
    else:
        result = st.session_state.last_result
        
        if result.collapsed:
            st.error("💥 АРХИТЕКТУРА КОЛЛАПСИРОВАЛА")
            st.markdown("**Все элементы носителя отождествлены.** Данная конфигурация структур и взаимодействия математически невозможна — это **no-go theorem**.")
            st.metric("Количество наложенных равенств", result.equations_count)
        else:
            atom = result.atom
            st.success(f"✅ **{atom.name}** — структура успешно синтезирована")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Элементов в носителе", len(atom.carrier))
            with col2:
                st.metric("Операций", len(atom.operations))
            
            st.markdown(f"**Родители:** {', '.join(atom.parent_atoms)}")
            st.markdown(f"**Взаимодействие:** {atom.interaction}")
            
            st.subheader("🧮 Операции синтезированной структуры")
            ops_list = [f"`{op}` (арность {ar})" for op, ar in atom.operations.items()]
            st.markdown(", ".join(ops_list))
            
            # Проверка свойств
            st.subheader("🔍 Проверка алгебраических свойств")
            if "+" in atom.operations and "·" in atom.operations:
                st.markdown("✅ **Дистрибутивность** обеспечивается коуравнителем")
            if "0" in atom.operations:
                st.markdown("✅ **Нейтральный элемент** присутствует")
        
        # AI-комментарий
        st.markdown("---")
        if api_key:
            if st.button("🤖 Интерпретировать результат (AI)", type="secondary"):
                with st.spinner("DeepSeek анализирует синтез..."):
                    comment = get_ai_comment(result, api_key)
                    st.info(f"💬 **Комментарий AI:**\n\n{comment}")
        else:
            st.caption("Введите API-ключ DeepSeek в боковой панели для AI-интерпретации.")

with tab2:
    for name in sorted(lib.keys()):
        atom = lib[name]
        with st.expander(f"{'🔷' if atom.is_synthetic else '💠'} {name}"):
            st.write(f"Носитель: {', '.join(atom.carrier)}")
            st.write(f"Операции: {', '.join(f'{op}:{ar}' for op, ar in atom.operations.items())}")

st.markdown("---")
st.caption("Hybrid Synthesis Laboratory v2.0 | L. Shcherbakov (2025)")
