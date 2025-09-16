# tests/test_add_header.py
from utils.patient import Patient

p = Patient("P1")
p.get_relevant_columns()   # primeira execução (adiciona ou corrige)
p.get_relevant_columns()   # segunda execução (não deve duplicar nem alterar)
print("Concluído")