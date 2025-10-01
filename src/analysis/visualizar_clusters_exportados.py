"""
Visualização de Clusters Exportados

Este script implementa o processo reverso da análise de clustering:
- Lê arquivos Excel de clusters exportados individualmente
- Reconstrói o dataset completo combinando os clusters
- Gera visualizações interativas dos clusters reconstruídos
- Permite análise comparativa entre clusters

Funcionalidades principais:
- Leitura automática de arquivos cluster_*.xlsx
- Reconstrução inteligente do dataset original
- Visualizações com PCA para redução dimensional
- Análise estatística por cluster
- Comparação visual entre clusters
- Nomenclatura diferenciada para evitar conflitos

Casos de uso:
1. Validação de clusters exportados
2. Análise visual após processamento externo
3. Verificação de integridade dos dados exportados
4. Geração de relatórios visuais para stakeholders

Estrutura de saída:
- clusters_exportados_overview.html: Visão geral de todos os clusters
- cluster_exportado_N.html: Visualização individual de cada cluster
- clusters_exportados_comparacao.html: Comparação entre clusters

Autor: Sistema de Análise Inteligente
Data: Outubro 2025
Versão: 1.0 - Visualização de clusters exportados
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
from pathlib import Path
import glob


class VisualizadorClustersExportados:
    """
    Classe para visualização e análise de clusters exportados individualmente.
    
    Esta classe implementa a funcionalidade reversa do pipeline de clustering:
    ao invés de partir dos dados brutos, ela parte dos clusters já exportados
    e reconstrói as visualizações e análises.
    
    Principais funcionalidades:
    - Carregamento automático de arquivos cluster_*.xlsx
    - Reconstrução do dataset original
    - Normalização consistente com análise original
    - Visualizações com PCA para interpretabilidade
    - Análises estatísticas por cluster
    - Comparações visuais entre clusters
    
    Attributes:
        pasta_clusters (str): Diretório onde estão os arquivos de clusters
        clusters_data (dict): Dicionário com dados de cada cluster {id: DataFrame}
        df_completo (pd.DataFrame): Dataset reconstruído completo
        df_scaled (pd.DataFrame): Dataset normalizado para visualizações
        scaler (StandardScaler): Objeto para normalização dos dados
        
    Example:
        >>> visualizador = VisualizadorClustersExportados()
        >>> resultados = visualizador.executar_visualizacao_completa()
        >>> print(f"Clusters processados: {resultados['clusters_carregados']}")
    """
    
    def __init__(self, pasta_clusters="outputs/data/"):
        """
        Inicializa o visualizador de clusters exportados.
        
        Args:
            pasta_clusters (str): Caminho para a pasta contendo os arquivos
                                cluster_*.xlsx. Default: "outputs/data/"
        """
        # Configurações de diretório
        self.pasta_clusters = pasta_clusters
        
        # Estruturas de dados
        self.clusters_data = {}     # {cluster_id: DataFrame}
        self.df_completo = None     # Dataset reconstruído
        self.df_scaled = None       # Dataset normalizado
        self.scaler = None          # Scaler para normalização
        
        # Criar diretório de figuras se não existir
        Path("outputs/figuras").mkdir(parents=True, exist_ok=True)
    
    def carregar_clusters_exportados(self):
        """
        Carrega automaticamente todos os arquivos de clusters exportados.
        
        Este método:
        1. Busca arquivos com padrão 'cluster_*.xlsx' na pasta especificada
        2. Extrai o ID do cluster do nome do arquivo
        3. Carrega a aba 'dados' de cada arquivo
        4. Adiciona coluna cluster_id para reconstruir o dataset
        5. Reconstrói o dataframe completo
        
        O processo é robusto e trata:
        - Arquivos em qualquer ordem
        - Diferentes números de clusters
        - Validação de estrutura dos arquivos
        
        Returns:
            dict: Dicionário com dados de cada cluster {cluster_id: DataFrame}
            
        Raises:
            FileNotFoundError: Se nenhum arquivo cluster_*.xlsx for encontrado
            ValueError: Se houver problema na estrutura dos arquivos
        """
        print("📂 Carregando clusters exportados...")
        
        # Buscar todos os arquivos cluster_*.xlsx na pasta
        pattern = os.path.join(self.pasta_clusters, "cluster_*.xlsx")
        arquivos_clusters = glob.glob(pattern)
        
        # Validar se encontrou arquivos
        if not arquivos_clusters:
            raise FileNotFoundError(f"Nenhum arquivo de cluster encontrado em {self.pasta_clusters}")
        
        # Carregar cada arquivo de cluster
        for arquivo in sorted(arquivos_clusters):
            # Extrair ID do cluster do nome do arquivo (cluster_X.xlsx -> X)
            nome_arquivo = os.path.basename(arquivo)
            try:
                cluster_id = int(nome_arquivo.split('_')[1].split('.')[0])
            except (IndexError, ValueError) as e:
                print(f"⚠️ Arquivo ignorado (nome inválido): {nome_arquivo}")
                continue
            
            try:
                # Carregar dados do cluster (aba 'dados')
                df_cluster = pd.read_excel(arquivo, sheet_name='dados')
                
                # Adicionar coluna cluster_id para reconstruir o dataset original
                df_cluster['cluster_id'] = cluster_id
                
                # Armazenar dados do cluster
                self.clusters_data[cluster_id] = df_cluster
                print(f"  ✅ Cluster {cluster_id}: {len(df_cluster)} registros")
                
            except Exception as e:
                print(f"⚠️ Erro ao carregar {arquivo}: {str(e)}")
                continue
        
        # Reconstruir dataframe completo a partir dos clusters
        self._reconstruir_dataframe_completo()
        
        print(f"\n📊 Total: {len(self.clusters_data)} clusters carregados")
        print(f"📋 Registros totais: {len(self.df_completo)}")
        
        return self.clusters_data
    
    def _reconstruir_dataframe_completo(self):
        """Reconstruir o dataframe completo a partir dos clusters"""
        # Concatenar todos os clusters
        dfs_clusters = list(self.clusters_data.values())
        self.df_completo = pd.concat(dfs_clusters, ignore_index=True)
        
        # Normalizar dados (excluindo cluster_id)
        features = [col for col in self.df_completo.columns if col != 'cluster_id']
        
        self.scaler = StandardScaler()
        dados_normalizados = self.scaler.fit_transform(self.df_completo[features])
        
        self.df_scaled = pd.DataFrame(
            dados_normalizados,
            columns=features,
            index=self.df_completo.index
        )
    
    def estatisticas_clusters_exportados(self):
        """Exibir estatísticas dos clusters carregados"""
        print("\n📈 Estatísticas dos Clusters Exportados:")
        print("=" * 50)
        
        for cluster_id, df_cluster in self.clusters_data.items():
            print(f"\n🔹 CLUSTER {cluster_id}:")
            print(f"   Registros: {len(df_cluster)-1}")  # -1 para excluir cluster_id column
            
            # Estatísticas das features (excluindo cluster_id)
            features = [col for col in df_cluster.columns if col != 'cluster_id']
            stats = df_cluster[features].describe()
            print(f"   Média geral: {df_cluster[features].mean().mean():.3f}")
            print(f"   Desvio padrão médio: {df_cluster[features].std().mean():.3f}")
        
        # Distribuição geral
        distribuicao = self.df_completo['cluster_id'].value_counts().sort_index()
        print(f"\n📊 Distribuição dos clusters:")
        for cluster_id, count in distribuicao.items():
            percentual = (count / len(self.df_completo)) * 100
            print(f"   Cluster {cluster_id}: {count} registros ({percentual:.1f}%)")
    
    def criar_visualizacao_overview_exportados(self):
        """Criar visualização overview dos clusters exportados"""
        print("\n🎨 Criando visualização overview dos clusters exportados...")
        
        # PCA para redução dimensional
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(self.df_scaled)
        
        # DataFrame para plotting
        df_plot = pd.DataFrame({
            'PC1': coords_2d[:, 0],
            'PC2': coords_2d[:, 1],
            'cluster': self.df_completo['cluster_id'].astype(str)
        })
        
        # Criar plot
        fig = px.scatter(
            df_plot, 
            x='PC1', 
            y='PC2', 
            color='cluster',
            title=f"Clusters Exportados - Visualização Reconstruída (k={len(self.clusters_data)})",
            labels={'cluster': 'Cluster ID'},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        
        # Adicionar informações no gráfico
        fig.add_annotation(
            text=f"Total: {len(self.df_completo)} registros<br>"
                 f"Clusters: {len(self.clusters_data)}<br>"
                 f"PCA Variance: {pca.explained_variance_ratio_.sum():.1%}",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.update_layout(
            height=700,
            width=900
        )
        
        # Salvar
        arquivo_saida = "outputs/figuras/clusters_exportados_overview.html"
        fig.write_html(arquivo_saida)
        fig.show()
        
        print(f"  ✅ Salvo: {arquivo_saida}")
        
        return fig
    
    def criar_visualizacoes_individuais_exportados(self):
        """Criar visualizações individuais para cada cluster exportado"""
        print("\n🎨 Criando visualizações individuais dos clusters exportados...")
        
        # PCA para todas as visualizações
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(self.df_scaled)
        
        figuras_criadas = []
        
        for cluster_id, df_cluster in self.clusters_data.items():
            # Máscara para este cluster
            mask = self.df_completo['cluster_id'] == cluster_id
            cluster_coords = coords_2d[mask]
            
            # Criar figura individual
            fig = go.Figure()
            
            # Adicionar pontos do cluster
            fig.add_trace(go.Scatter(
                x=cluster_coords[:, 0],
                y=cluster_coords[:, 1],
                mode='markers',
                name=f'Cluster {cluster_id}',
                marker=dict(
                    size=8,
                    color=px.colors.qualitative.Set1[cluster_id % len(px.colors.qualitative.Set1)],
                    opacity=0.7
                ),
                text=[f'Ponto {i}' for i in range(len(cluster_coords))],
                hovertemplate='<b>Cluster %{customdata}</b><br>' +
                             'PC1: %{x:.3f}<br>' +
                             'PC2: %{y:.3f}<br>' +
                             '<extra></extra>',
                customdata=[cluster_id] * len(cluster_coords)
            ))
            
            # Calcular estatísticas do cluster
            features = [col for col in df_cluster.columns if col != 'cluster_id']
            media_geral = df_cluster[features].mean().mean()
            
            # Layout
            fig.update_layout(
                title=f"Cluster {cluster_id} Exportado - Visualização Individual",
                xaxis_title="Componente Principal 1",
                yaxis_title="Componente Principal 2",
                height=600,
                width=800,
                showlegend=False
            )
            
            # Adicionar informações do cluster
            fig.add_annotation(
                text=f"<b>Cluster {cluster_id}</b><br>"
                     f"Registros: {len(df_cluster)-1}<br>"  # -1 para cluster_id
                     f"Média Geral: {media_geral:.3f}<br>"
                     f"% do Total: {(len(df_cluster)/len(self.df_completo))*100:.1f}%",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor=px.colors.qualitative.Set1[cluster_id % len(px.colors.qualitative.Set1)],
                borderwidth=2
            )
            
            # Salvar figura
            arquivo_saida = f"outputs/figuras/cluster_exportado_{cluster_id}.html"
            fig.write_html(arquivo_saida)
            fig.show()
            
            figuras_criadas.append(arquivo_saida)
            print(f"  ✅ Salvo: {arquivo_saida}")
        
        return figuras_criadas
    
    def criar_comparacao_clusters_exportados(self):
        """Criar visualização comparativa dos clusters por features"""
        print("\n🎨 Criando comparação entre clusters exportados...")
        
        # Preparar dados para comparação
        features = [col for col in self.df_completo.columns if col != 'cluster_id']
        n_features = len(features)
        
        # Calcular médias por cluster
        medias_clusters = {}
        for cluster_id in self.clusters_data.keys():
            mask = self.df_completo['cluster_id'] == cluster_id
            medias_clusters[cluster_id] = self.df_completo[mask][features].mean()
        
        # Criar subplots
        cols = min(3, n_features)
        rows = (n_features + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=features,
            specs=[[{"type": "bar"}] * cols for _ in range(rows)]
        )
        
        colors = px.colors.qualitative.Set1
        
        # Adicionar gráficos de barras para cada feature
        for i, feature in enumerate(features):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            cluster_ids = list(self.clusters_data.keys())
            valores = [medias_clusters[cid][feature] for cid in cluster_ids]
            
            fig.add_trace(
                go.Bar(
                    x=[f'Cluster {cid}' for cid in cluster_ids],
                    y=valores,
                    name=feature,
                    marker_color=[colors[cid % len(colors)] for cid in cluster_ids],
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Comparação de Features entre Clusters Exportados",
            height=300 * rows,
            showlegend=False
        )
        
        # Salvar
        arquivo_saida = "outputs/figuras/clusters_exportados_comparacao.html"
        fig.write_html(arquivo_saida)
        fig.show()
        
        print(f"  ✅ Salvo: {arquivo_saida}")
        
        return fig
    
    def executar_visualizacao_completa(self):
        """Executar todo o pipeline de visualização dos clusters exportados"""
        print("🚀 Iniciando Visualização de Clusters Exportados")
        print("=" * 60)
        
        try:
            # 1. Carregar clusters
            self.carregar_clusters_exportados()
            
            # 2. Mostrar estatísticas
            self.estatisticas_clusters_exportados()
            
            # 3. Visualização overview
            self.criar_visualizacao_overview_exportados()
            
            # 4. Visualizações individuais
            figuras_individuais = self.criar_visualizacoes_individuais_exportados()
            
            # 5. Comparação entre clusters
            self.criar_comparacao_clusters_exportados()
            
            print("\n" + "=" * 60)
            print("✅ Visualização dos clusters exportados concluída!")
            print(f"📁 Figuras salvas em: outputs/figuras/")
            print(f"📊 {len(figuras_individuais)} visualizações individuais criadas")
            print("=" * 60)
            
            return {
                'clusters_carregados': len(self.clusters_data),
                'total_registros': len(self.df_completo),
                'figuras_criadas': len(figuras_individuais) + 2  # +2 para overview e comparação
            }
            
        except Exception as e:
            print(f"❌ Erro durante a visualização: {str(e)}")
            raise


def main():
    """Função principal para execução do script"""
    visualizador = VisualizadorClustersExportados()
    resultados = visualizador.executar_visualizacao_completa()
    return resultados


if __name__ == "__main__":
    # Verificar se pandas está disponível
    try:
        import pandas as pd
        print("✅ Ambiente detectado - executando visualização...")
        main()
    except ImportError:
        print("❌ Pandas não encontrado. Ative o ambiente virtual primeiro:")
        print("source .env")
        import sys
        sys.exit(1)