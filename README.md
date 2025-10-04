# Biblioteca de Árvores de Decisão (ID3, C4.5, CART)

Esta é uma biblioteca educacional em Python com implementações dos algoritmos de **árvores de decisão** mais conhecidos:  
- ID3  
- C4.5  
- CART

## Instalação

Para instalar faça:

```bash
pip install git+https://github.com/Sofia-0812/bibliotecaArvoresDeDecisao.git
```

## Requisitos
- Python 3.8+
- numpy
- pandas
- graphviz

## Utilização
```bash
from mytrees import ID3DecisionTreeClassifier
from mytrees import C45DecisionTreeClassifier
from mytrees import CARTDecisionTreeClassifier

# Exemplo CART
clf = CARTDecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
```
