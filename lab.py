"""
Hybrid Synthesis Laboratory — Полная версия
Расширенная библиотека: группы, кольца, решётки, категории, кватернионы, векторные пространства.
"""

import streamlit as st
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import itertools
from typing import List, Tuple, Dict, Set, Optional, Callable
import json
import copy
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════
# CORE ENGINE (без изменений)
# ═══════════════════════════════════════════════════════════════════

class Term:
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
    def close(self, equations: List[Tuple[Term, Term]], arities: Dict[str, int]):
    """Замыкание: начальное объединение + распространение конгруэнтности (безопасная версия)."""
    # Шаг 1: Объединяем все равенства
    for left, right in equations:
        self.union(left, right)
    
    # Шаг 2: Ограниченное распространение — только на известные термы
    # Не генерируем новые комбинации, только проверяем существующие
    changed = True
    max_iterations = 100  # защита от бесконечного цикла
    iteration = 0
    
    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        
        # Собираем все известные термы
        all_terms = list(self.parent.keys())
        
        # Для каждой операции проверяем, не появились ли новые равенства
        for op, arity in arities.items():
            if arity == 0:
                continue
            
            # Перебираем существующие термы, а не все возможные комбинации
            for t in all_terms:
                if t.head != op:
                    continue
                if len(t.args) != arity:
                    continue
                
                # Для каждого аргумента проверяем, равен ли он чему-то ещё
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

    def close(self, equations: List[Tuple[Term, Term]], arities: Dict[str, int]):
        for left, right in equations:
            self.union(left, right)

        changed = True
        while changed:
            changed = False
            all_terms = list(self.parent.keys())
            for op, arity in arities.items():
                if arity == 0:
                    continue
                for combo in itertools.product(all_terms, repeat=arity):
                    t = Term(op, list(combo))
                    for i in range(arity):
                        for other in all_terms:
                            if self.find(combo[i]) == self.find(other):
                                new_combo = list(combo)
                                new_combo[i] = other
                                if self.union(t, Term(op, new_combo)):
                                    changed = True


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

    def all_ground_terms(self, extra_ops: Dict[str, int] = None) -> Set[Term]:
        terms = {Term(el) for el in self.carrier}
        all_ops = dict(self.operations)
        if extra_ops:
            all_ops.update(extra_ops)
        for op, arity in all_ops.items():
            if arity == 0:
                terms.add(Term(op, []))
        return terms


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

    terms = set()
    terms.update({Term(el) for el in A.carrier})
    terms.update({Term(el) for el in B.carrier})

    for op, arity in all_ops.items():
        if arity == 0:
            terms.add(Term(op, []))

    equations: List[Tuple[Term, Term]] = []

    for left, right in A.axioms:
        vars_left = left.variables()
        vars_right = right.variables()
        all_vars = vars_left | vars_right
        for combo in itertools.product(A.carrier, repeat=len(all_vars)):
            mapping = {var: Term(val) for var, val in zip(sorted(all_vars), combo)}
            equations.append((left.substitute(mapping), right.substitute(mapping)))

    for left_b, right_b in B.axioms:
        vars_left = left_b.variables()
        vars_right = right_b.variables()
        all_vars_b = vars_left | vars_right
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

    cc = CongruenceClosure()
    cc.close(equations, all_ops)

    classes = defaultdict(list)
    for t in list(cc.parent.keys()):
        classes[cc.find(t)].append(t)

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

    new_axioms = []
    for (left, right) in equations[:50]:
        new_axioms.append((left, right))

    new_atom = Atom(
        name=f"{A.name}⊕{B.name}_{action_name}",
        carrier=new_carrier,
        operations=new_operations,
        axioms=new_axioms,
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
# EXTENDED BUILT-IN LIBRARY
# ═══════════════════════════════════════════════════════════════════

def create_builtin_library() -> Dict[str, Atom]:
    lib = {}

    # ── ГРУППЫ ──────────────────────────────────────────────
    
    # Z2 аддитивная
    Z2 = Atom(
        name="Z₂ (additive group)",
        carrier=["0", "1"],
        operations={"+": 2, "0": 0, "-": 1},
        axioms=[
            (Term("+", [Term("x"), Term("0")]), Term("x")),
            (Term("+", [Term("0"), Term("x")]), Term("x")),
            (Term("+", [Term("x"), Term("-", [Term("x")])]), Term("0")),
            (Term("+", [Term("x"), Term("y")]), Term("+", [Term("y"), Term("x")])),
            (Term("+", [Term("+", [Term("x"), Term("y")]), Term("z")]),
             Term("+", [Term("x"), Term("+", [Term("y"), Term("z")])])),
        ],
        description="Циклическая абелева группа порядка 2."
    )
    lib[Z2.name] = Z2

    # Z3 аддитивная
    Z3 = Atom(
        name="Z₃ (additive group)",
        carrier=["0", "1", "2"],
        operations={"+": 2, "0": 0, "-": 1},
        axioms=[],
        description="Циклическая абелева группа порядка 3."
    )
    add_Z3 = {
        ("0","0"):"0", ("0","1"):"1", ("0","2"):"2",
        ("1","0"):"1", ("1","1"):"2", ("1","2"):"0",
        ("2","0"):"2", ("2","1"):"0", ("2","2"):"1",
    }
    for (a,b), c in add_Z3.items():
        Z3.axioms.append((Term("+", [Term(a), Term(b)]), Term(c)))
    Z3.axioms.append((Term("-", [Term("0")]), Term("0")))
    Z3.axioms.append((Term("-", [Term("1")]), Term("2")))
    Z3.axioms.append((Term("-", [Term("2")]), Term("1")))
    lib[Z3.name] = Z3

    # Группа Клейна V4 (Z2 × Z2)
    V4 = Atom(
        name="Klein Four-Group V₄",
        carrier=["0", "a", "b", "c"],
        operations={"+": 2, "0": 0, "-": 1},
        axioms=[],
        description="Группа Клейна: абелева, каждый элемент — обратный себе."
    )
    # Таблица Кэли для V4
    add_V4 = {
        ("0","0"):"0", ("0","a"):"a", ("0","b"):"b", ("0","c"):"c",
        ("a","0"):"a", ("a","a"):"0", ("a","b"):"c", ("a","c"):"b",
        ("b","0"):"b", ("b","a"):"c", ("b","b"):"0", ("b","c"):"a",
        ("c","0"):"c", ("c","a"):"b", ("c","b"):"a", ("c","c"):"0",
    }
    for (x, y), z in add_V4.items():
        V4.axioms.append((Term("+", [Term(x), Term(y)]), Term(z)))
    V4.axioms.append((Term("-", [Term("0")]), Term("0")))
    V4.axioms.append((Term("-", [Term("a")]), Term("a")))
    V4.axioms.append((Term("-", [Term("b")]), Term("b")))
    V4.axioms.append((Term("-", [Term("c")]), Term("c")))
    lib[V4.name] = V4

    # Тривиальная группа
    Triv = Atom(
        name="Trivial Group {e}",
        carrier=["e"],
        operations={"+": 2, "0": 0, "-": 1},
        axioms=[
            (Term("+", [Term("e"), Term("e")]), Term("e")),
        ],
        description="Тривиальная группа из одного элемента."
    )
    lib[Triv.name] = Triv

    # S3 (симметрическая группа на 3 элементах)
    S3 = Atom(
        name="S₃ (Symmetric Group)",
        carrier=["id", "a", "b", "c", "d", "f"],  # 6 перестановок
        operations={"∘": 2, "id": 0, "inv": 1},
        axioms=[],  # заполним упрощённо — как группа перестановок
        description="Симметрическая группа порядка 6 (неабелева)."
    )
    # Упрощённая таблица Кэли для S3 (используем generators: a=(12), b=(23))
    # Здесь мы просто задаём структуру с носителем и операциями без полной таблицы
    # В реальной песочнице нужно заполнить все 36 комбинаций
    lib[S3.name] = S3

    # ── МОНОИДЫ ────────────────────────────────────────────
    
    M2 = Atom(
        name="Z₂ (multiplicative monoid)",
        carrier=["0", "1"],
        operations={"·": 2, "1": 0},
        axioms=[
            (Term("·", [Term("x"), Term("1")]), Term("x")),
            (Term("·", [Term("1"), Term("x")]), Term("x")),
            (Term("·", [Term("0"), Term("x")]), Term("0")),
            (Term("·", [Term("x"), Term("0")]), Term("0")),
            (Term("·", [Term("·", [Term("x"), Term("y")]), Term("z")]),
             Term("·", [Term("x"), Term("·", [Term("y"), Term("z")])])),
        ],
        description="Мультипликативный моноид {0,1}."
    )
    lib[M2.name] = M2

    TrivMon = Atom(
        name="Trivial Monoid",
        carrier=["e"],
        operations={"·": 2, "e": 0},
        axioms=[(Term("·", [Term("e"), Term("e")]), Term("e"))],
        description="Тривиальный моноид."
    )
    lib[TrivMon.name] = TrivMon

    # ── ПОЛУГРУППЫ ────────────────────────────────────────
    
    # Циклическая полугруппа порядка 3
    Sem3 = Atom(
        name="Cyclic Semigroup order 3",
        carrier=["a", "a²", "a³"],
        operations={"*": 2},
        axioms=[
            (Term("*", [Term("a"), Term("a")]), Term("a²")),
            (Term("*", [Term("a"), Term("a²")]), Term("a³")),
            (Term("*", [Term("a²"), Term("a")]), Term("a³")),
            (Term("*", [Term("a"), Term("a³")]), Term("a³")),
            (Term("*", [Term("a³"), Term("a")]), Term("a³")),
            (Term("*", [Term("a²"), Term("a²")]), Term("a³")),
            (Term("*", [Term("a²"), Term("a³")]), Term("a³")),
            (Term("*", [Term("a³"), Term("a²")]), Term("a³")),
            (Term("*", [Term("a³"), Term("a³")]), Term("a³")),
        ],
        description="Циклическая полугруппа: a → a² → a³ (поглощающий)."
    )
    lib[Sem3.name] = Sem3

    # ── КОЛЬЦА ─────────────────────────────────────────────
    
    Z2_ring = Atom(
        name="Z₂ (ring)",
        carrier=["0", "1"],
        operations={"+": 2, "0": 0, "-": 1, "·": 2, "1": 0},
        axioms=[],
        description="Кольцо/поле Z₂."
    )
    lib[Z2_ring.name] = Z2_ring

    Z4_ring = Atom(
        name="Z₄ (ring, not a field)",
        carrier=["0", "1", "2", "3"],
        operations={"+": 2, "0": 0, "-": 1, "·": 2, "1": 0},
        axioms=[],
        description="Кольцо Z₄ (2·2=0, есть делители нуля)."
    )
    lib[Z4_ring.name] = Z4_ring

    # ── РЕШЁТКИ ───────────────────────────────────────────
    
    Bool = Atom(
        name="Boolean Lattice {0,1}",
        carrier=["0", "1"],
        operations={"∧": 2, "∨": 2, "¬": 1, "0": 0, "1": 0},
        axioms=[
            (Term("∧", [Term("x"), Term("x")]), Term("x")),
            (Term("∨", [Term("x"), Term("x")]), Term("x")),
            (Term("∧", [Term("x"), Term("y")]), Term("∧", [Term("y"), Term("x")])),
            (Term("∨", [Term("x"), Term("y")]), Term("∨", [Term("y"), Term("x")])),
            (Term("∧", [Term("x"), Term("0")]), Term("0")),
            (Term("∨", [Term("x"), Term("1")]), Term("1")),
            (Term("∧", [Term("x"), Term("¬", [Term("x")])]), Term("0")),
            (Term("∨", [Term("x"), Term("¬", [Term("x")])]), Term("1")),
        ],
        description="Булева алгебра на {0,1}."
    )
    lib[Bool.name] = Bool

    # Дистрибутивная решётка без дополнения (M3)
    M3_lattice = Atom(
        name="M₃ Lattice (no complement)",
        carrier=["⊥", "a", "b", "c", "⊤"],
        operations={"∧": 2, "∨": 2},
        axioms=[
            (Term("∧", [Term("x"), Term("x")]), Term("x")),
            (Term("∨", [Term("x"), Term("x")]), Term("x")),
            (Term("∧", [Term("x"), Term("y")]), Term("∧", [Term("y"), Term("x")])),
            (Term("∨", [Term("x"), Term("y")]), Term("∨", [Term("y"), Term("x")])),
            (Term("∧", [Term("x"), Term("⊥")]), Term("⊥")),
            (Term("∨", [Term("x"), Term("⊤")]), Term("⊤")),
            (Term("∧", [Term("x"), Term("⊤")]), Term("x")),
            (Term("∨", [Term("x"), Term("⊥")]), Term("x")),
            # a,b,c несравнимы
            (Term("∧", [Term("a"), Term("b")]), Term("⊥")),
            (Term("∧", [Term("b"), Term("c")]), Term("⊥")),
            (Term("∧", [Term("a"), Term("c")]), Term("⊥")),
            (Term("∨", [Term("a"), Term("b")]), Term("⊤")),
            (Term("∨", [Term("b"), Term("c")]), Term("⊤")),
            (Term("∨", [Term("a"), Term("c")]), Term("⊤")),
        ],
        description="Дистрибутивная решётка M₃ (не булева)."
    )
    lib[M3_lattice.name] = M3_lattice

    # ── ВЕКТОРНЫЕ ПРОСТРАНСТВА ──────────────────────────
    
    V2_GF2 = Atom(
        name="V₂ over GF(2)",
        carrier=["0", "e1", "e2", "e1+e2"],
        operations={"+": 2, "0": 0, "-": 1},
        axioms=[],  # будем уточнять
        description="Двумерное векторное пространство над полем из двух элементов."
    )
    add_V2 = {
        ("0","0"):"0", ("0","e1"):"e1", ("0","e2"):"e2", ("0","e1+e2"):"e1+e2",
        ("e1","0"):"e1", ("e1","e1"):"0", ("e1","e2"):"e1+e2", ("e1","e1+e2"):"e2",
        ("e2","0"):"e2", ("e2","e1"):"e1+e2", ("e2","e2"):"0", ("e2","e1+e2"):"e1",
        ("e1+e2","0"):"e1+e2", ("e1+e2","e1"):"e2", ("e1+e2","e2"):"e1", ("e1+e2","e1+e2"):"0",
    }
    for (x,y), z in add_V2.items():
        V2_GF2.axioms.append((Term("+", [Term(x), Term(y)]), Term(z)))
    lib[V2_GF2.name] = V2_GF2

    # ── КВАТЕРНИОНЫ НАД Z₂ ───────────────────────────────
    
    Quat_Z2 = Atom(
        name="Quaternions over Z₂",
        carrier=["0", "1", "i", "j", "k", "1+i", "1+j", "1+k", "i+j", "i+k", "j+k", "1+i+j", "1+i+k", "1+j+k", "i+j+k", "1+i+j+k"],
        operations={"+": 2, "0": 0, "*": 2, "1": 0},
        axioms=[],  # заглушка — полная таблица Кэли потребовала бы 256 строк
        description="Алгебра кватернионов над Z₂ (некоммутативная)."
    )
    lib[Quat_Z2.name] = Quat_Z2

    # ── КАТЕГОРИИ И ГРАФЫ ──────────────────────────────
    
    FreeCat = Atom(
        name="Free Category on 2-object graph",
        carrier=["id_A", "id_B", "f", "g", "g∘f"],
        operations={"∘": 2, "id": 0},
        axioms=[
            (Term("∘", [Term("id_A"), Term("f")]), Term("f")),
            (Term("∘", [Term("f"), Term("id_A")]), Term("f")),
            (Term("∘", [Term("id_B"), Term("g")]), Term("g")),
            (Term("∘", [Term("g"), Term("id_B")]), Term("g")),
            (Term("∘", [Term("g"), Term("f")]), Term("g∘f")),
        ],
        description="Свободная категория на графе с 2 объектами и одной композицией."
    )
    lib[FreeCat.name] = FreeCat

    # ── ВЕРОЯТНОСТНЫЕ ПРОСТРАНСТВА ─────────────────────
    
    Coin = Atom(
        name="Fair Coin (H,T)",
        carrier=["H", "T"],
        operations={},
        axioms=[],
        description="Вероятностное пространство с двумя исходами."
    )
    lib[Coin.name] = Coin

    Omega3 = Atom(
        name="Ω = {1,2,3}",
        carrier=["ω1", "ω2", "ω3"],
        operations={},
        axioms=[],
        description="Внешний индекс для случайности."
    )
    lib[Omega3.name] = Omega3

    # ── КВАЗИГРУППЫ И ЛУПЫ ──────────────────────────────
    
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

    # Лупа Муфанг порядка 4 (минимальная неассоциативная лупа)
    Moufang4 = Atom(
        name="Moufang Loop order 4",
        carrier=["e", "a", "b", "c"],
        operations={"*": 2, "e": 0},
        axioms=[
            # Единица
            (Term("*", [Term("e"), Term("x")]), Term("x")),
            (Term("*", [Term("x"), Term("e")]), Term("x")),
            # Таблица умножения (неассоциативная)
            (Term("*", [Term("a"), Term("a")]), Term("e")),
            (Term("*", [Term("b"), Term("b")]), Term("e")),
            (Term("*", [Term("c"), Term("c")]), Term("e")),
            (Term("*", [Term("a"), Term("b")]), Term("c")),
            (Term("*", [Term("b"), Term("c")]), Term("a")),
            (Term("*", [Term("c"), Term("a")]), Term("b")),
            (Term("*", [Term("b"), Term("a")]), Term("c")),
            (Term("*", [Term("c"), Term("b")]), Term("a")),
            (Term("*", [Term("a"), Term("c")]), Term("b")),
        ],
        description="Лупа Муфанг порядка 4 (неассоциативная: a*(b*c) ≠ (a*b)*c)."
    )
    lib[Moufang4.name] = Moufang4

    return lib


# ═══════════════════════════════════════════════════════════════════
# STREAMLIT UI (как раньше, без изменений)
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Hybrid Synthesis Lab", page_icon="🧬", layout="wide")
st.title("🧬 Hybrid Synthesis Laboratory")
st.markdown("**Полноценная лаборатория архитектурного синтеза алгебраических структур**")
st.caption("Библиотека: группы, кольца, решётки, категории, векторные пространства, кватернионы, полугруппы, лупы.")

# Состояние
if 'library' not in st.session_state:
    st.session_state.library = create_builtin_library()

if 'synthesis_history' not in st.session_state:
    st.session_state.synthesis_history = []

# Боковая панель
with st.sidebar:
    st.header("📚 Библиотека")
    lib = st.session_state.library
    names = sorted(lib.keys())
    filter_text = st.text_input("🔍 Поиск", "")
    if filter_text:
        names = [n for n in names if filter_text.lower() in n.lower()]
    st.caption(f"Структур: {len(names)}")

    st.subheader("Синтез")
    atom_a_name = st.selectbox("Атом A (цель)", names, key="a")
    atom_b_name = st.selectbox("Атом B (оператор)", names, key="b")
    action_name = st.text_input("Действие", "·")

    if st.button("🚀 Синтезировать", type="primary", use_container_width=True):
        A = lib[atom_a_name]
        B = lib[atom_b_name]
        with st.spinner(f"Синтез {A.name} ⊕ {B.name}..."):
            result = synthesize(A, B, action_name)
        st.session_state.last_result = result
        if result.collapsed:
            st.error("💥 Коллапс!")
            st.session_state.synthesis_history.append({
                "atoms": (atom_a_name, atom_b_name),
                "action": action_name,
                "collapsed": True,
                "timestamp": result.timestamp
            })
        else:
            st.success(f"✅ **{result.atom.name}**")
            st.balloons()
            lib[result.atom.name] = result.atom
            st.session_state.synthesis_history.append({
                "atoms": (atom_a_name, atom_b_name),
                "action": action_name,
                "collapsed": False,
                "result_name": result.atom.name,
                "timestamp": result.timestamp
            })

    st.markdown("---")
    st.subheader("🛠️ Новый атом")
    new_name = st.text_input("Название")
    new_carrier = st.text_input("Носитель (через запятую)")
    new_ops = st.text_input("Операции (имя:арность)")
    if st.button("Добавить"):
        try:
            carrier = [x.strip() for x in new_carrier.split(",") if x.strip()]
            ops = {}
            for part in new_ops.split(","):
                if ":" in part:
                    k, v = part.strip().split(":")
                    ops[k.strip()] = int(v.strip())
            lib[new_name] = Atom(name=new_name, carrier=carrier, operations=ops, description="Пользовательский.")
            st.success(f"Атом '{new_name}' добавлен!")
            st.rerun()
        except Exception as e:
            st.error(f"Ошибка: {e}")

# Вкладки
tab1, tab2, tab3 = st.tabs(["🔬 Результат", "📖 Библиотека", "📜 История"])

with tab1:
    if 'last_result' not in st.session_state:
        st.info("Выполните синтез.")
    else:
        result = st.session_state.last_result
        if result.collapsed:
            st.error("💥 Архитектура коллапсировала.")
            st.metric("Равенств", result.equations_count)
        else:
            atom = result.atom
            st.success(f"✅ **{atom.name}**")
            st.metric("Носитель", len(atom.carrier))
            st.metric("Операций", len(atom.operations))
            st.markdown(f"**Родители:** {', '.join(atom.parent_atoms)}")
            with st.expander("Классы эквивалентности"):
                for rep, elems in sorted(result.classes.items(), key=lambda x: repr(x[0])):
                    st.write(f"**{repr(rep)}** → {{{', '.join(map(repr, elems[:10]))}}}")

with tab2:
    st.header("📖 Все структуры")
    for name in sorted(lib.keys()):
        atom = lib[name]
        with st.expander(f"{'🔷' if atom.is_synthetic else '💠'} {name}"):
            st.markdown(f"**Носитель:** {', '.join(atom.carrier)}")
            st.markdown(f"**Операции:** {', '.join(f'{op}:{ar}' for op, ar in atom.operations.items())}")
            if atom.description:
                st.caption(atom.description)

with tab3:
    st.header("📜 История")
    if not st.session_state.synthesis_history:
        st.info("Пусто.")
    else:
        for entry in reversed(st.session_state.synthesis_history):
            if entry["collapsed"]:
                st.error(f"{entry['atoms'][0]} ⊕ {entry['atoms'][1]} → 💥")
            else:
                st.success(f"{entry['atoms'][0]} ⊕ {entry['atoms'][1]} → **{entry['result_name']}**")

st.markdown("---")
st.caption("Hybrid Synthesis Laboratory v2.0 | E. Azari & L. Shcherbakov (2025)")
