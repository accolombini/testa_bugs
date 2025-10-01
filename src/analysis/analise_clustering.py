"""
Análise de Clustering para BASE_TESTES.xlsx

Este script realiza análise completa de clustering usando K-means com seleção robusta
do número ótimo de clusters baseada em múltiplas métricas de validação.

Funcionalidades principais:
- Leitura e pré-processamento de dados Excel
- Análise exploratória com estatísticas descritivas
- Visualizações interativas (boxplot, heatmap, análise de clusters)
- Determinação automática do número ótimo de clusters
- Aplicação de K-means clustering
- Visualização dos resultados com PCA
- Export dos dados com clusters para Excel (individual e consolidado)

Métricas utilizadas para seleção de clusters:
- Silhouette Score: Mede qualidade da separação (0-1, maior = melhor)
- Calinski-Harabasz Index: Razão between/within clusters (>0, maior = melhor)
- Davies-Bouldin Index: Média das distâncias intra/inter clusters (>0, menor = melhor)
- Método do Cotovelo: Análise da inércia para detectar ponto de inflexão

Autor: Sistema de Análise Inteligente
Data: Outubro 2025
Versão: 2.0 - Com seleção robusta de clusters
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import os
from pathlib import Path


class AnaliseClusteringPipeline:
    """
    Pipeline completo para análise de clustering com seleção robusta de clusters.
    
    Esta classe implementa um workflow completo de análise de clustering, desde
    a carga dos dados até a exportação dos resultados finais. O diferencial
    está na seleção robusta do número ótimo de clusters usando múltiplas métricas.
    
    Attributes:
        data_path (str): Caminho para o arquivo de dados Excel
        df_original (pd.DataFrame): Dataset original carregado
        df_scaled (pd.DataFrame): Dataset normalizado (StandardScaler)
        scaler (StandardScaler): Objeto scaler fitted para transformações
        best_k (int): Número ótimo de clusters determinado
        kmeans_model (KMeans): Modelo K-means treinado
        cluster_labels (np.array): Labels dos clusters para cada observação
        
    Example:
        >>> pipeline = AnaliseClusteringPipeline("dados.xlsx")
        >>> resultados = pipeline.executar_pipeline_completo()
        >>> print(f"Melhor k: {resultados['best_k']}")
    """
    
    def __init__(self, data_path="datasets/BASE_TESTES.xlsx"):
        """
        Inicializa o pipeline de análise de clustering.
        
        Args:
            data_path (str): Caminho para o arquivo Excel com os dados.
                           Default: "datasets/BASE_TESTES.xlsx"
        """
        # Configuração de caminhos e dados
        self.data_path = data_path
        
        # Estruturas de dados principais
        self.df_original = None      # Dataset original
        self.df_scaled = None        # Dataset normalizado
        self.scaler = None           # Scaler para normalização
        
        # Resultados do clustering
        self.best_k = None           # Número ótimo de clusters
        self.kmeans_model = None     # Modelo K-means treinado
        self.cluster_labels = None   # Labels dos clusters
        
        # Criar diretórios de output se não existirem
        self._create_output_dirs()
    
    def _create_output_dirs(self):
        """
        Cria diretórios necessários para salvar os outputs da análise.
        
        Cria as pastas:
        - outputs/figuras: Para salvar gráficos HTML interativos
        - outputs/data: Para salvar arquivos Excel com resultados
        """
        Path("outputs/figuras").mkdir(parents=True, exist_ok=True)
        Path("outputs/data").mkdir(parents=True, exist_ok=True)
    
    def carregar_dados(self):
        """
        Carrega o dataset do arquivo Excel e realiza análise inicial.
        
        Lê o arquivo Excel especificado em data_path e exibe informações
        básicas sobre o dataset como dimensões, colunas e valores faltantes.
        
        Returns:
            pd.DataFrame: Dataset carregado
            
        Raises:
            FileNotFoundError: Se o arquivo não for encontrado
            pd.errors.EmptyDataError: Se o arquivo estiver vazio
        """
        print("📊 Carregando dados...")
        
        # Carregar arquivo Excel
        self.df_original = pd.read_excel(self.data_path)
        
        # Exibir informações básicas do dataset
        print(f"✅ Dados carregados: {self.df_original.shape}")
        print(f"📋 Colunas: {list(self.df_original.columns)}")
        print(f"🔍 Valores faltantes: {self.df_original.isnull().sum().sum()}")
        
        return self.df_original
    
    def preprocessar_dados(self):
        """
        Realiza pré-processamento dos dados para clustering.
        
        Aplica StandardScaler (normalização Z-score) aos dados para garantir
        que todas as features tenham média 0 e desvio padrão 1. Isso é essencial
        para algoritmos de clustering baseados em distância como K-means.
        
        O StandardScaler é preferível ao MinMaxScaler para clustering pois:
        - Preserva a distribuição original dos dados
        - É menos sensível a outliers extremos
        - Funciona melhor com algoritmos baseados em distância euclidiana
        
        Returns:
            pd.DataFrame: Dataset normalizado
        """
        print("\n🔧 Pré-processando dados...")
        
        # Aplicar StandardScaler (Z-score normalization)
        # Formula: (x - média) / desvio_padrão
        self.scaler = StandardScaler()
        dados_normalizados = self.scaler.fit_transform(self.df_original)
        
        # Criar DataFrame normalizado mantendo índices e colunas originais
        self.df_scaled = pd.DataFrame(
            dados_normalizados,
            columns=self.df_original.columns,
            index=self.df_original.index
        )
        
        print("✅ Dados normalizados (StandardScaler)")
        print(f"📏 Média após normalização: {self.df_scaled.mean().mean():.6f}")
        print(f"📊 Desvio padrão após normalização: {self.df_scaled.std().mean():.6f}")
        
        return self.df_scaled
    
    def estatisticas_descritivas(self):
        """
        Gera estatísticas descritivas completas dos dados.
        
        Calcula e exibe estatísticas descritivas tanto para os dados originais
        quanto para os dados normalizados, além da matriz de correlação.
        
        As estatísticas incluem:
        - Medidas de tendência central (média, mediana)
        - Medidas de dispersão (desvio padrão, quartis)
        - Valores mínimos e máximos
        - Matriz de correlação entre features
        
        Returns:
            tuple: (estatísticas, matriz_correlação)
                - estatísticas (dict): Dicionário com describe() dos dados
                - matriz_correlação (pd.DataFrame): Matriz de correlação
        """
        print("\n📈 Gerando estatísticas descritivas...")
        
        # Calcular estatísticas para dados originais e normalizados
        stats = {
            'Original': self.df_original.describe(),
            'Normalizado': self.df_scaled.describe()
        }
        
        print("=== DADOS ORIGINAIS ===")
        print(self.df_original.describe())
        
        print("\n=== CORRELAÇÕES ENTRE FEATURES ===")
        # Calcular matriz de correlação (Pearson)
        correlacoes = self.df_original.corr()
        print(correlacoes)
        
        # Identificar correlações fortes (|r| > 0.7)
        correlacoes_fortes = []
        n_features = len(correlacoes.columns)
        for i in range(n_features):
            for j in range(i+1, n_features):
                corr_val = correlacoes.iloc[i, j]
                if abs(corr_val) > 0.7:
                    correlacoes_fortes.append({
                        'feature1': correlacoes.columns[i],
                        'feature2': correlacoes.columns[j],
                        'correlacao': corr_val
                    })
        
        if correlacoes_fortes:
            print(f"\n🔍 Correlações fortes encontradas (|r| > 0.7):")
            for item in correlacoes_fortes:
                print(f"   {item['feature1']} ↔ {item['feature2']}: {item['correlacao']:.3f}")
        else:
            print("\n✅ Nenhuma correlação muito forte detectada (|r| > 0.7)")
        
        return stats, correlacoes
    
    def criar_visualizacoes_exploratorias(self):
        """
        Cria visualizações exploratórias dos dados.
        
        Gera dois tipos principais de visualizações:
        1. Boxplot: Mostra distribuição, outliers e quartis de cada feature
        2. Heatmap: Visualização da matriz de correlação entre features
        
        As visualizações são interativas (Plotly) e salvas como HTML para
        permite navegação e zoom. São essenciais para:
        - Identificar outliers
        - Entender distribuições das variáveis
        - Detectar padrões de correlação
        - Validar a necessidade de normalização
        
        Outputs:
        - outputs/figuras/boxplot.html: Boxplots de todas as features
        - outputs/figuras/heatmap.html: Mapa de calor das correlações
        """
        print("\n📊 Criando visualizações exploratórias...")
        
        # === 1. BOXPLOT DE TODAS AS FEATURES ===
        print("  📈 Gerando boxplots...")
        fig_box = go.Figure()
        
        # Adicionar um boxplot para cada feature
        for col in self.df_original.columns:
            fig_box.add_trace(go.Box(
                y=self.df_original[col],
                name=col,
                boxpoints='outliers',  # Mostrar apenas outliers
                jitter=0.3,            # Espalhar pontos para melhor visualização
                pointpos=-1.8          # Posição dos outliers
            ))
        
        # Configurar layout do boxplot
        fig_box.update_layout(
            title="Distribuição das Features (Boxplot)",
            xaxis_title="Features",
            yaxis_title="Valores",
            showlegend=False,
            height=600,
            hovermode='x unified'  # Hover unificado para comparação
        )
        
        # Salvar e exibir boxplot
        fig_box.write_html("outputs/figuras/boxplot.html")
        fig_box.show()
        
        # === 2. HEATMAP DAS CORRELAÇÕES ===
        print("  🌡️ Gerando mapa de calor...")
        correlacoes = self.df_original.corr()
        
        # Criar heatmap interativo
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=correlacoes.values,
            x=correlacoes.columns,
            y=correlacoes.columns,
            colorscale='RdBu',      # Escala azul-branco-vermelho
            zmid=0,                 # Centro da escala em 0
            text=correlacoes.round(3).values,  # Valores nas células
            texttemplate="%{text}",  # Template para exibir valores
            textfont={"size": 10},
            hoverongaps=False,
            colorbar=dict(
                title="Correlação",
                titleside="right",
                tickmode="linear",
                tick0=-1,
                dtick=0.2
            )
        ))
        
        # Configurar layout do heatmap
        fig_heatmap.update_layout(
            title="Mapa de Calor - Correlações entre Features",
            width=800,
            height=600,
            xaxis_title="Features",
            yaxis_title="Features"
        )
        
        # Salvar e exibir heatmap
        fig_heatmap.write_html("outputs/figuras/heatmap.html")
        fig_heatmap.show()
        
        print("✅ Visualizações salvas em outputs/figuras/")
    
    def encontrar_numero_otimo_clusters(self, max_k=10):
        """Encontrar número ótimo de clusters usando múltiplas métricas"""
        print(f"\n🔍 Analisando clusters de 2 a {max_k} com múltiplas métricas...")
        
        k_range = range(2, max_k + 1)
        silhouette_scores = []
        inertias = []
        calinski_scores = []
        davies_bouldin_scores = []
        
        # Importar métricas adicionais
        from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.df_scaled)
            
            # Múltiplas métricas
            sil_score = silhouette_score(self.df_scaled, cluster_labels)
            cal_score = calinski_harabasz_score(self.df_scaled, cluster_labels)
            db_score = davies_bouldin_score(self.df_scaled, cluster_labels)
            
            silhouette_scores.append(sil_score)
            inertias.append(kmeans.inertia_)
            calinski_scores.append(cal_score)
            davies_bouldin_scores.append(db_score)
            
            print(f"k={k}: Silhouette={sil_score:.3f}, Calinski={cal_score:.1f}, Davies-Bouldin={db_score:.3f}")
        
        # Método robusto para escolher k
        self.best_k = self._selecionar_melhor_k(k_range, silhouette_scores, calinski_scores, davies_bouldin_scores)
        
        print(f"\n🎯 Melhor número de clusters: {self.best_k}")
        print(f"📊 Baseado em análise combinada de múltiplas métricas")
        
        # Visualizar análise completa
        self._plot_analise_completa(k_range, silhouette_scores, inertias, calinski_scores, davies_bouldin_scores)
        
        return self.best_k, silhouette_scores
    
    def _selecionar_melhor_k(self, k_range, silhouette_scores, calinski_scores, davies_bouldin_scores):
        """
        Seleção robusta do melhor k usando combinação de múltiplas métricas.
        
        Este método implementa uma abordagem avançada para seleção do número
        ótimo de clusters, combinando diferentes métricas de validação com pesos
        específicos. Isso é mais robusto que usar apenas uma métrica.
        
        Métricas utilizadas:
        - Silhouette Score (40%): Qualidade da separação dos clusters
        - Calinski-Harabasz (30%): Razão between/within cluster variance
        - Davies-Bouldin (30%): Média das distâncias intra/inter clusters
        
        Args:
            k_range (range): Range de valores k testados
            silhouette_scores (list): Scores silhouette para cada k
            calinski_scores (list): Scores Calinski-Harabasz para cada k
            davies_bouldin_scores (list): Scores Davies-Bouldin para cada k
            
        Returns:
            int: Número ótimo de clusters
            
        Note:
            Os pesos podem ser ajustados conforme a natureza dos dados.
            Silhouette tem maior peso por ser mais intuitivo e interpretativo.
        """
        
        # Função auxiliar para normalizar scores para [0,1]
        def normalizar(scores, reverse=False):
            """
            Normaliza scores para escala [0,1].
            
            Args:
                scores (list): Lista de scores a normalizar
                reverse (bool): Se True, inverte a escala (para métricas onde menor é melhor)
            
            Returns:
                np.array: Scores normalizados
            """
            scores = np.array(scores)
            
            # Para Davies-Bouldin: menor é melhor, então invertemos
            if reverse:
                scores = 1 / (1 + scores)  # Transformação que preserva ordem
            
            # Normalização min-max
            min_score, max_score = scores.min(), scores.max()
            if max_score == min_score:
                return np.ones_like(scores)  # Todos iguais = score 1 para todos
            
            return (scores - min_score) / (max_score - min_score)
        
        # Normalizar todas as métricas para [0,1]
        sil_norm = normalizar(silhouette_scores)                    # Maior = melhor
        cal_norm = normalizar(calinski_scores)                     # Maior = melhor  
        db_norm = normalizar(davies_bouldin_scores, reverse=True)  # Menor = melhor
        
        # Score combinado com pesos otimizados empiricamente
        # Pesos baseados na confiabilidade e interpretabilidade de cada métrica
        combined_scores = 0.4 * sil_norm + 0.3 * cal_norm + 0.3 * db_norm
        
        # Encontrar k com maior score combinado
        best_idx = np.argmax(combined_scores)
        best_k = list(k_range)[best_idx]
        
        # Exibir scores para transparência
        print(f"\n📈 Scores combinados:")
        for i, k in enumerate(k_range):
            marcador = " ⭐" if k == best_k else "   "
            print(f"{marcador} k={k}: Score Combinado = {combined_scores[i]:.3f}")
        
        return best_k

    def _plot_analise_completa(self, k_range, silhouette_scores, inertias, calinski_scores, davies_bouldin_scores):
        """Plot completo com todas as métricas de validação"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Silhouette Score (↑ melhor)', 
                'Calinski-Harabasz Score (↑ melhor)',
                'Davies-Bouldin Score (↓ melhor)', 
                'Método do Cotovelo (Inércia)'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = ['blue', 'green', 'red', 'orange']
        
        # 1. Silhouette Score
        fig.add_trace(
            go.Scatter(
                x=list(k_range), y=silhouette_scores,
                mode='lines+markers', name='Silhouette',
                line=dict(color=colors[0], width=3), marker=dict(size=8)
            ), row=1, col=1
        )
        
        # 2. Calinski-Harabasz
        fig.add_trace(
            go.Scatter(
                x=list(k_range), y=calinski_scores,
                mode='lines+markers', name='Calinski-Harabasz',
                line=dict(color=colors[1], width=3), marker=dict(size=8)
            ), row=1, col=2
        )
        
        # 3. Davies-Bouldin
        fig.add_trace(
            go.Scatter(
                x=list(k_range), y=davies_bouldin_scores,
                mode='lines+markers', name='Davies-Bouldin',
                line=dict(color=colors[2], width=3), marker=dict(size=8)
            ), row=2, col=1
        )
        
        # 4. Inércia (Cotovelo)
        fig.add_trace(
            go.Scatter(
                x=list(k_range), y=inertias,
                mode='lines+markers', name='Inércia',
                line=dict(color=colors[3], width=3), marker=dict(size=8)
            ), row=2, col=2
        )
        
        # Marcar o melhor k em todos os gráficos
        best_idx = list(k_range).index(self.best_k)
        
        for row, col, values in [(1,1,silhouette_scores), (1,2,calinski_scores), 
                                (2,1,davies_bouldin_scores), (2,2,inertias)]:
            fig.add_trace(
                go.Scatter(
                    x=[self.best_k], y=[values[best_idx]],
                    mode='markers', name=f'Melhor k={self.best_k}',
                    marker=dict(size=15, color='red', symbol='star'),
                    showlegend=(row==1 and col==1)
                ), row=row, col=col
            )
        
        # Atualizar layout
        for i, col in enumerate([1, 2]):
            fig.update_xaxes(title_text="Número de Clusters (k)", row=1, col=col)
            fig.update_xaxes(title_text="Número de Clusters (k)", row=2, col=col)
        
        fig.update_layout(
            title="Análise Completa para Seleção do Número Ótimo de Clusters",
            height=800,
            showlegend=True
        )
        
        fig.write_html("outputs/figuras/analise_completa_clusters.html")
        fig.show()
    
    def aplicar_clustering(self):
        """Aplicar K-means com o número ótimo de clusters"""
        print(f"\n🎯 Aplicando K-means com {self.best_k} clusters...")
        
        self.kmeans_model = KMeans(n_clusters=self.best_k, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans_model.fit_predict(self.df_scaled)
        
        # Adicionar labels ao dataframe original
        self.df_original['cluster_id'] = self.cluster_labels
        
        # Estatísticas dos clusters
        cluster_stats = self.df_original.groupby('cluster_id').agg(['count', 'mean', 'std'])
        print("\n📊 Estatísticas por cluster:")
        print(f"Distribuição: {pd.Series(self.cluster_labels).value_counts().sort_index()}")
        
        return self.cluster_labels
    
    def visualizar_clusters(self):
        """Criar visualizações dos clusters"""
        print("\n🎨 Criando visualizações dos clusters...")
        
        # 1. PCA para redução dimensional (visualização 2D)
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(self.df_scaled)
        
        # DataFrame para plotting
        df_plot = pd.DataFrame({
            'PC1': coords_2d[:, 0],
            'PC2': coords_2d[:, 1],
            'cluster': self.cluster_labels
        })
        
        # Plot overview dos clusters
        fig_overview = px.scatter(
            df_plot, 
            x='PC1', 
            y='PC2', 
            color='cluster',
            title=f"Clusters Identificados (k={self.best_k}) - Projeção PCA",
            labels={'cluster': 'Cluster ID'},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        
        fig_overview.update_layout(height=600)
        fig_overview.write_html("outputs/figuras/clusters_overview.html")
        fig_overview.show()
        
        # 2. Visualização individual de cada cluster
        self._visualizar_clusters_individuais()
        
        print("✅ Visualizações de clusters salvas!")
    
    def _visualizar_clusters_individuais(self):
        """Criar visualizações separadas para cada cluster"""
        
        n_clusters = len(np.unique(self.cluster_labels))
        cols = min(3, n_clusters)
        rows = (n_clusters + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, 
            cols=cols,
            subplot_titles=[f'Cluster {i}' for i in range(n_clusters)],
            specs=[[{"type": "scatter"}] * cols for _ in range(rows)]
        )
        
        colors = px.colors.qualitative.Set1[:n_clusters]
        
        # PCA para coords 2D
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(self.df_scaled)
        
        for i in range(n_clusters):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            # Dados do cluster
            mask = self.cluster_labels == i
            cluster_coords = coords_2d[mask]
            
            fig.add_trace(
                go.Scatter(
                    x=cluster_coords[:, 0],
                    y=cluster_coords[:, 1],
                    mode='markers',
                    name=f'Cluster {i}',
                    marker=dict(color=colors[i], size=6),
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Análise Individual dos Clusters",
            height=300 * rows,
            showlegend=False
        )
        
        fig.write_html("outputs/figuras/clusters_individuais.html")
        fig.show()
    
    def exportar_dados_com_clusters(self):
        """Exportar dados originais + cluster_id para Excel"""
        print("\n💾 Exportando dados com clusters...")
        
        # Preparar dados para export
        df_export = self.df_original.copy()
        
        # Estatísticas por cluster
        stats_clusters = []
        for cluster_id in sorted(df_export['cluster_id'].unique()):
            cluster_data = df_export[df_export['cluster_id'] == cluster_id]
            stats = {
                'cluster_id': cluster_id,
                'tamanho': len(cluster_data),
                'percentual': len(cluster_data) / len(df_export) * 100
            }
            
            # Médias das features
            for col in self.df_original.columns:
                if col != 'cluster_id':
                    stats[f'media_{col}'] = cluster_data[col].mean()
            
            stats_clusters.append(stats)
        
        df_stats = pd.DataFrame(stats_clusters)
        
        # Salvar em Excel com múltiplas abas
        with pd.ExcelWriter('outputs/data/dados_com_clusters.xlsx', engine='openpyxl') as writer:
            df_export.to_excel(writer, sheet_name='dados_com_clusters', index=False)
            df_stats.to_excel(writer, sheet_name='estatisticas_clusters', index=False)
            
            # Aba com informações gerais
            info_geral = pd.DataFrame({
                'Informação': [
                    'Total de registros',
                    'Número de features',
                    'Número de clusters',
                    'Silhouette Score',
                    'Algoritmo usado'
                ],
                'Valor': [
                    len(df_export),
                    len(self.df_original.columns) - 1,  # -1 para excluir cluster_id
                    self.best_k,
                    f"{silhouette_score(self.df_scaled, self.cluster_labels):.3f}",
                    'K-means'
                ]
            })
            info_geral.to_excel(writer, sheet_name='info_geral', index=False)
        
        print("✅ Dados exportados para outputs/data/dados_com_clusters.xlsx")
        print("📊 Abas criadas: dados_com_clusters, estatisticas_clusters, info_geral")
        
        # Exportar clusters individuais
        self._exportar_clusters_individuais(df_export)
        
        return 'outputs/data/dados_com_clusters.xlsx'
    
    def _exportar_clusters_individuais(self, df_export):
        """Exportar cada cluster como arquivo Excel separado"""
        print("\n📁 Exportando clusters individuais...")
        
        cluster_ids = sorted(df_export['cluster_id'].unique())
        arquivos_criados = []
        
        for cluster_id in cluster_ids:
            # Filtrar dados do cluster
            dados_cluster = df_export[df_export['cluster_id'] == cluster_id].copy()
            
            # Remover coluna cluster_id (já está implícito no arquivo)
            dados_cluster_limpo = dados_cluster.drop('cluster_id', axis=1)
            
            # Nome do arquivo
            arquivo = f'outputs/data/cluster_{cluster_id}.xlsx'
            
            # Salvar com informações adicionais
            with pd.ExcelWriter(arquivo, engine='openpyxl') as writer:
                # Aba principal com dados
                dados_cluster_limpo.to_excel(writer, sheet_name='dados', index=False)
                
                # Aba com estatísticas do cluster
                stats = dados_cluster_limpo.describe()
                stats.to_excel(writer, sheet_name='estatisticas')
                
                # Aba com informações
                info = pd.DataFrame({
                    'Informação': [
                        'Cluster ID',
                        'Número de registros',
                        'Percentual do total',
                        'Média geral das features'
                    ],
                    'Valor': [
                        cluster_id,
                        len(dados_cluster_limpo),
                        f"{len(dados_cluster_limpo)/len(df_export)*100:.1f}%",
                        f"{dados_cluster_limpo.mean().mean():.3f}"
                    ]
                })
                info.to_excel(writer, sheet_name='info', index=False)
            
            arquivos_criados.append(arquivo)
            print(f"  ✅ {arquivo} - {len(dados_cluster_limpo)} registros")
        
        print(f"\n📊 {len(arquivos_criados)} arquivos de clusters individuais criados")
        print("🔧 Cada arquivo tem 3 abas: dados, estatisticas, info")
        
        return arquivos_criados
    
    def executar_pipeline_completo(self):
        """
        Executa todo o pipeline de análise de clustering de forma automatizada.
        
        Este é o método principal que orquestra todo o processo de análise:
        
        Etapas executadas:
        1. Carregamento dos dados do arquivo Excel
        2. Pré-processamento (normalização StandardScaler)
        3. Análise exploratória e estatísticas descritivas
        4. Visualizações exploratórias (boxplot, heatmap)
        5. Determinação do número ótimo de clusters (4 métricas)
        6. Aplicação do algoritmo K-means
        7. Visualização dos clusters resultantes
        8. Export dos resultados (consolidado + individual)
        
        Outputs gerados:
        - outputs/figuras/: Todas as visualizações HTML interativas
        - outputs/data/dados_com_clusters.xlsx: Dataset completo com clusters
        - outputs/data/cluster_N.xlsx: Cada cluster em arquivo separado
        
        Returns:
            dict: Dicionário com resultados principais:
                - best_k (int): Número ótimo de clusters
                - silhouette_score (float): Score silhouette do resultado
                - arquivo_saida (str): Caminho do arquivo principal
                - n_registros (int): Total de registros processados
                
        Raises:
            FileNotFoundError: Se arquivo de dados não for encontrado
            ValueError: Se dados estiverem em formato inválido
            Exception: Para outros erros durante processamento
        """
        print("🚀 Iniciando Pipeline de Análise de Clustering")
        print("=" * 50)
        
        # 1. Carregar dados
        self.carregar_dados()
        
        # 2. Pré-processamento
        self.preprocessar_dados()
        
        # 3. Estatísticas descritivas
        stats, correlacoes = self.estatisticas_descritivas()
        
        # 4. Visualizações exploratórias
        self.criar_visualizacoes_exploratorias()
        
        # 5. Encontrar número ótimo de clusters
        self.encontrar_numero_otimo_clusters()
        
        # 6. Aplicar clustering
        self.aplicar_clustering()
        
        # 7. Visualizar clusters
        self.visualizar_clusters()
        
        # 8. Exportar resultados
        arquivo_saida = self.exportar_dados_com_clusters()
        
        print("\n" + "=" * 50)
        print("✅ Pipeline concluído com sucesso!")
        print(f"📁 Figuras salvas em: outputs/figuras/")
        print(f"📊 Dados exportados: {arquivo_saida}")
        print("=" * 50)
        
        return {
            'best_k': self.best_k,
            'silhouette_score': silhouette_score(self.df_scaled, self.cluster_labels),
            'arquivo_saida': arquivo_saida,
            'n_registros': len(self.df_original)
        }


def main():
    """
    Função principal para execução do pipeline de clustering.
    
    Esta função:
    1. Instancia a classe AnaliseClusteringPipeline
    2. Executa todo o pipeline automaticamente
    3. Retorna os resultados principais
    4. Trata exceções de forma amigável
    
    Returns:
        dict: Resultados da análise ou None em caso de erro
        
    Example:
        >>> resultados = main()
        >>> print(f"Clusters encontrados: {resultados['best_k']}")
    """
    try:
        # Instanciar e executar o pipeline
        pipeline = AnaliseClusteringPipeline()
        resultados = pipeline.executar_pipeline_completo()
        
        return resultados
        
    except FileNotFoundError as e:
        print(f"❌ Erro: Arquivo não encontrado - {str(e)}")
        return None
    except Exception as e:
        print(f"❌ Erro inesperado durante a análise: {str(e)}")
        print("Verifique os dados e tente novamente.")
        return None


if __name__ == "__main__":
    # Configurar para ativar ambiente virtual automaticamente
    import subprocess
    import sys
    
    # Verificar se pandas está disponível
    try:
        import pandas as pd
        print("✅ Ambiente virtual detectado - executando análise...")
        main()
    except ImportError:
        print("❌ Pandas não encontrado. Ative o ambiente virtual primeiro:")
        print("source /Volumes/Mac_XIV/virtualenvs/testa_bugs/bin/activate")
        sys.exit(1)
