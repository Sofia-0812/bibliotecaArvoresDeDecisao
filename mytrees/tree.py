import pandas as pd
import numpy as np
import graphviz

def calculate_entropy(data):
    labels = pd.Series(data).value_counts()
    total = len(data)
    if total == 0:
        return 0.0
    entropy = -sum((count / total) * np.log2(count / total) for count in labels)
    return entropy

def calculate_information_gain(X, y, attribute):
    total_entropy = calculate_entropy(y)
    values = X[attribute].unique()
    weighted_entropy = 0.0
    n = len(y)
    if n == 0:
        return 0.0
    for value in values:
        subset_y = y[X[attribute] == value]
        weighted_entropy += (len(subset_y) / n) * calculate_entropy(subset_y)
    return total_entropy - weighted_entropy

def split_info(group_sizes):
    total = sum(group_sizes)
    if total == 0:
        return 0.0
    probs = [g/total for g in group_sizes if g > 0]
    return -sum(p * np.log2(p) for p in probs)

def calculate_gini(data):
    y = pd.Series(data)
    total = len(data)
    if total == 0:
        return 0.0
    counts = data.value_counts()
    probs = counts / total
    return 1.0 - np.sum(probs ** 2)

class BaseDecisionTreeClassifier:
    """Classe base unificada para todas as árvores de decisão (ID3, C4.5, CART)"""

    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
        self.feature_names_ = None
        self.default_class_ = None
        self.algorithm = None

    def fit(self, X, y):
        # 1. Normalização de entrada (DataFrame e Series)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        y = pd.Series(y).reset_index(drop=True)
        X = X.reset_index(drop=True)

        # 2. Configurações iniciais
        self.feature_names_ = list(X.columns)
        self.default_class_ = pd.Series(y).mode().iat[0]

        # 3. Construção da árvore
        self.tree = self._build_tree(X, y, depth=0)
        return self

    # --- Funções de Seleção de Atributos ---

    def _find_best_id3_split(self, X, y, features):
        """ID3: Information Gain simples (apenas atributos categóricos)"""
        # Filtra apenas colunas categóricas/object (como ID3 exige)
        gains = {f: calculate_information_gain(X, y, f) for f in features if X[f].dtype == 'object'}
        if not gains:
            return None, None

        # Encontra a pontuação máxima
        max_gain = max(gains.values())

        # Encontra todos os atributos que empatam com a pontuação máxima
        tied_features = [f for f, gain in gains.items() if gain == max_gain]

        # CRITÉRIO DE DESEMPATE: Escolhe o atributo de nome menor (ordem alfabética)
        best_feature = min(tied_features)

        return best_feature, None # ID3/Categorical sempre tem threshold=None para indicar multiway split


    def _find_best_c45_split(self, X, y, features):
        """C4.5: Gain Ratio (atributos categóricos split multiway, atributos numéricos split binário)"""
        best_feature = None
        best_threshold = None
        best_gr = -float('inf')

        # Lista para armazenar todos os splits ótimos em caso de empate (Feature, Threshold, Score)
        optimal_splits = []

        for f in features:
            if X[f].dtype == 'object':  # Categórico - multiway
                ig = calculate_information_gain(X, y, f)
                counts = [len(y[X[f] == v]) for v in X[f].unique()]
                si = split_info(counts)
                gr = ig / si if si > 0 else 0.0
                # Armazena o split se for o melhor até agora ou se empatar
                if gr > best_gr:
                    best_gr = gr
                    optimal_splits = [(f, None, gr)] # Novo melhor: limpa e adiciona
                elif gr == best_gr and gr > 0:
                    optimal_splits.append((f, None, gr)) # Empate: adiciona

            else:  # Numérica - binary
                gr, thr = self._best_continuous_split(X, y, f, 'C45')

                if gr > best_gr:
                    best_gr = gr
                    optimal_splits = [(f, thr, gr)]
                elif gr == best_gr and gr > 0:
                    optimal_splits.append((f, thr, gr))

        if not optimal_splits:
            return None, None

        # Encontra a pontuação máxima (apenas para garantir)
        max_score = max(s[2] for s in optimal_splits)

        # Filtra todos que alcançaram o score máximo (pode ser redundante, mas seguro)
        final_ties = [(f, t) for f, t, score in optimal_splits if score == max_score]

        # CRITÉRIO DE DESEMPATE: Ordena alfabeticamente e escolhe o primeiro
        best_feature, best_threshold = min(final_ties, key=lambda x: x[0])

        return best_feature, best_threshold

    def _find_best_cart_split(self, X, y, features):
        """CART: Gini decrease (apenas atributos numéricos e aplits binários)"""
        best_feature = None
        best_threshold = None
        best_decrease = -float('inf')

        optimal_splits = []

        for f in features:
            if X[f].dtype != 'object':  # Numérica - binary
                decrease, thr = self._best_continuous_split(X, y, f, 'CART')

                if decrease > best_decrease:
                    best_decrease = decrease
                    optimal_splits = [(f, thr, decrease)]
                elif decrease == best_decrease and decrease > 0:
                    optimal_splits.append((f, thr, decrease))

        if not optimal_splits:
            return None, None

        # Encontra a pontuação máxima
        max_score = max(s[2] for s in optimal_splits)

        # Filtra todos que alcançaram o score máximo
        final_ties = [(f, t) for f, t, score in optimal_splits if score == max_score]

        # CRITÉRIO DE DESEMPATE: Ordena alfabeticamente e escolhe o primeiro
        # Usa min com a chave (key) para ordenar pelo nome da feature (índice 0)
        best_feature, best_threshold = min(final_ties, key=lambda x: x[0])

        return best_feature, best_threshold

    def _best_continuous_split(self, X, y, feature, algorithm):
        """Split contínuo com midpoints"""
        col = X[feature]
        uniq = np.sort(col.unique())

        # Cria midpoints como candidatos a threshold
        if len(uniq) < 2:
             return (0.0, None) # Não há split possível

        candidates = (uniq[:-1] + uniq[1:]) / 2.0
        best_score = -float('inf')
        best_thr = None

        for thr in candidates:
            mask = col < thr
            left_y, right_y = y[mask], y[~mask]

            if len(left_y) == 0 or len(right_y) == 0:
                continue

            n = len(y)

            if algorithm == 'C45':
                # Cálculo do Gain Ratio
                total_entropy = calculate_entropy(y)
                weighted = (len(left_y)/n) * calculate_entropy(left_y) + (len(right_y)/n) * calculate_entropy(right_y)
                ig = total_entropy - weighted
                si = split_info([len(left_y), len(right_y)])
                score = ig / si if si > 0 else 0.0
            else:  # CART
                # Cálculo do Gini decrease (diminuição da impureza)
                parent_gini = calculate_gini(y)
                weighted_gini = (len(left_y)/n) * calculate_gini(left_y) + (len(right_y)/n) * calculate_gini(right_y)
                score = parent_gini - weighted_gini

            if score > best_score:
                best_score = score
                best_thr = thr

        # Retorna 0.0 (C4.5) ou -inf (CART) se nenhum split foi encontrado
        if best_thr is None:
            return (0.0, None) if algorithm == 'C45' else (-float('inf'), None)

        return best_score, best_thr

    # --- Funções de Construção da Árvore ---

    def _build_tree(self, X, y, depth):
        # 1. Casos base
        # Se todos os exemplos são da mesma classe OU conjunto de dados está vazio
        if len(y) == 0 or len(pd.Series(y).unique()) == 1:
            return pd.Series(y).mode().iat[0] if len(y) > 0 else self.default_class_
        # Profundidade máxima atingida
        if self.max_depth is not None and depth >= self.max_depth:
            return pd.Series(y).mode().iat[0]
        # Não há mais features para dividir
        if len(X.columns) == 0:
             return pd.Series(y).mode().iat[0]

        # 2. Encontra o melhor split
        features = list(X.columns)

        if self.algorithm == 'ID3':
            best_feature, best_threshold = self._find_best_id3_split(X, y, features)
        elif self.algorithm == 'C45':
            best_feature, best_threshold = self._find_best_c45_split(X, y, features)
        elif self.algorithm == 'CART':
            best_feature, best_threshold = self._find_best_cart_split(X, y, features)
        else:
            raise ValueError(f"Algoritmo desconhecido: {self.algorithm}")


        # Retorna nó folha
        if best_feature is None:
            return pd.Series(y).mode().iat[0]

        # 3. Constrói o nó
        if best_threshold is None:  # Categórico (ID3, C4.5)
            # O ID3 só usa splits categóricos, então esta é a sua única via
            return self._build_categorical_node(X, y, depth, best_feature)
        else:  # Contínuo (C4.5, CART)
            return self._build_continuous_node(X, y, depth, best_feature, best_threshold)

    def _build_categorical_node(self, X, y, depth, best_feature):
        node = {'type': 'categorical', 'feature': best_feature, 'branches': {}}

        for value in X[best_feature].unique():
            mask = X[best_feature] == value
            subset_X = X[mask]
            subset_y = y[mask]

            # Se o subconjunto estiver vazio, retorna a classe de moda do nó pai
            if len(subset_y) == 0:
                node['branches'][value] = pd.Series(y).mode().iat[0]
            else:
                # Remove a coluna categórica para a próxima recursão, o mesmo atributo categórico não pode ser usado duas vezes.
                subset_X_filtered = subset_X.drop(columns=[best_feature])
                # Chama a recursão
                node['branches'][value] = self._build_tree(subset_X_filtered, subset_y, depth + 1)

        return node

    def _build_continuous_node(self, X, y, depth, best_feature, best_threshold):
        node = {'type': 'continuous', 'feature': best_feature, 'threshold': best_threshold}

        mask = X[best_feature] < best_threshold
        left_X, left_y = X[mask], y[mask]
        right_X, right_y = X[~mask], y[~mask]

        # Chamadas recursivas (o atributo contínuo NÃO é removido, pode ser usado novamente em outros splits)
        node['left'] = self._build_tree(left_X, left_y, depth + 1)
        node['right'] = self._build_tree(right_X, right_y, depth + 1)
        return node

    # --- Função de Predição ---

    def predict(self, X):
        """Faz predições para um DataFrame."""
        if self.tree is None:
            raise Exception("O classificador não foi treinado. Chame 'fit' primeiro.")

        if not isinstance(X, pd.DataFrame):
            # Tenta reconstruir o DataFrame usando os nomes de features do treinamento
            X = pd.DataFrame(X, columns=self.feature_names_)

        return np.array([self._predict_row(row, self.tree) for _, row in X.iterrows()])

    def _predict_row(self, row, node):
        """Navega recursivamente na árvore para fazer uma predição em uma linha."""
        # Se for um nó folha (classe de saída)
        if not isinstance(node, dict):
            return node

        # Nó de decisão
        feature = node['feature']

        if node['type'] == 'categorical':
            val = row[feature]
            if val not in node['branches']:
                return self.default_class_

            return self._predict_row(row, node['branches'][val])

        else:  # Nó do tipo contínuo
            if row[feature] < node['threshold']:
                return self._predict_row(row, node['left'])
            else:
                return self._predict_row(row, node['right'])


    def plot_tree(self, graph_size=None, image_width="100%"):
        if self.tree is None:
            raise Exception("A árvore não foi treinada. Chame 'fit' primeiro.")

        # 1. Configuração dos atributos do grafo
        graph_attrs = {
            'rankdir': 'TB',
            'splines': 'true',
            'bgcolor': 'transparent',
            'width': image_width
        }

        # Adiciona o tamanho do grafo se fornecido
        if graph_size:
          graph_attrs['size'] = graph_size
          graph_attrs['ratio'] = 'fill' # Ajuda a usar o tamanho definido

        # 1. Inicializa o objeto Digraph (grafo direcionado)
        dot = graphviz.Digraph(format='svg', graph_attr=graph_attrs)

        # Variável auxiliar para gerar IDs de nó exclusivos
        self._node_count = 0

        # 2. Chama a função auxiliar recursiva para construir o grafo
        self._plot_node(dot, self.tree)

        # 3. Retorna o objeto Digraph.
        return dot

    def _plot_node(self, dot, node, parent_id=None, edge_label=""):
        current_id = str(self._node_count)
        self._node_count += 1

        if not isinstance(node, dict):
            # Nó Folha (Terminal): Retângulo com o rótulo da classe
            label = str(node)
            dot.node(current_id, label, shape='box', style='filled', fillcolor='#D9FFD9', color='#006600') # Verde suave
        else:
            # Nó de Decisão: Oval com o atributo de split
            feature = node.get('feature')
            node_type = node.get('type')

            if node_type == 'categorical':
                dot.node(current_id, str(feature), shape='oval', style='filled', fillcolor='#B0E0E6', color='#008080') # Azul suave

                # Recursão para cada ramo categórico
                for value, subtree in node['branches'].items():
                    # O rótulo da aresta é o valor da feature
                    self._plot_node(dot, subtree, current_id, str(value))

            elif node_type == 'continuous':
                threshold = node.get('threshold')

                # Limpa a representação do float para melhor visualização
                display_threshold = f"{threshold:.2f}" if isinstance(threshold, (float, np.float64)) else str(threshold)

                dot.node(current_id, str(feature), shape='oval', style='filled', fillcolor='#FFFACD', color='#FFA500') # Amarelo suave

                # Ramo ESQUERDO (Condição: feature < threshold)
                left_label = f"< {display_threshold}"
                self._plot_node(dot, node['left'], current_id, left_label)

                # Ramo DIREITO (Condição: feature >= threshold)
                right_label = f">= {display_threshold}"
                self._plot_node(dot, node['right'], current_id, right_label)

            else:
                # Caso de nó de dicionário inválido ou incompleto
                dot.node(current_id, "ERROR", shape='box', style='filled', fillcolor='red')

        # Conecta este nó ao seu nó pai
        if parent_id is not None:
            dot.edge(parent_id, current_id, label=edge_label)

class ID3DecisionTreeClassifier(BaseDecisionTreeClassifier):
    def __init__(self, max_depth=None):
        super().__init__(max_depth)
        self.algorithm = 'ID3'

class C45DecisionTreeClassifier(BaseDecisionTreeClassifier):
    def __init__(self, max_depth=None):
        super().__init__(max_depth)
        self.algorithm = 'C45'

class CARTDecisionTreeClassifier(BaseDecisionTreeClassifier):
    def __init__(self, max_depth=None):
        super().__init__(max_depth)
        self.algorithm = 'CART'
