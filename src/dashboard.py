import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path

# Configuração da página


if __name__ == '__main__':
    st.set_page_config(
        page_title="High-Frequency Trading Analytics Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Título e instruções
    st.title("📊 High-Frequency Trading Analytics Dashboard")
    st.markdown("""
    ### Instruções de Uso:
    1. **Preview dos Dados**: Visualize as primeiras linhas e estatísticas descritivas
    2. **Análise Visual**: Explore gráficos interativos das features numéricas
    3. **Download**: Baixe os dados processados em formato CSV
    4. **Filtros**: Use a barra lateral para personalizar a visualização

    ---
    """)

    # Função para carregar dados
    @st.cache_data
    def load_data():
        """Carrega os dados do arquivo features.parquet"""
        try:
            data_path = Path("data/processed/features.parquet")
            if data_path.exists():
                df = pd.read_parquet(data_path)
                return df
            else:
                st.error(f"Arquivo não encontrado: {data_path}")
                return None
        except Exception as e:
            st.error(f"Erro ao carregar dados: {str(e)}")
            return None

    # Carregar dados
    df = load_data()

    if df is not None:
        # Sidebar para filtros
        st.sidebar.header("🔧 Configurações")

        # Informações básicas dos dados
        st.sidebar.markdown(f"**Total de registros:** {len(df):,}")
        st.sidebar.markdown(f"**Features disponíveis:** {len(df.columns)}")

        # Filtro de data (se existir coluna de timestamp)
        if 'timestamp' in df.columns or any('time' in col.lower() for col in df.columns):
            time_cols = [col for col in df.columns if 'time' in col.lower()]
            if time_cols:
                st.sidebar.subheader("📅 Filtro Temporal")
                # Aqui você pode adicionar filtros de data se necessário

        # Seleção de features numéricas para visualização
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if numeric_cols:
            st.sidebar.subheader("📈 Features para Visualização")
            selected_features = st.sidebar.multiselect(
                "Selecione as features numéricas:",
                options=numeric_cols,
                default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols
            )

        # Seção 1: Preview dos Dados
        st.header("📋 Preview dos Dados")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Primeiras 10 linhas")
            st.dataframe(df.head(10), use_container_width=True)

        with col2:
            st.subheader("Estatísticas Descritivas")
            st.dataframe(df.describe(), use_container_width=True)

        # Informações sobre tipos de dados
        st.subheader("ℹ️ Informações das Colunas")
        info_df = pd.DataFrame({
            'Coluna': df.columns,
            'Tipo': df.dtypes.astype(str),
            'Valores Únicos': [df[col].nunique() for col in df.columns],
            'Valores Nulos': [df[col].isnull().sum() for col in df.columns],
            '% Nulos': [f"{(df[col].isnull().sum() / len(df) * 100):.2f}%" for col in df.columns]
        })
        st.dataframe(info_df, use_container_width=True)

        # Seção 2: Visualizações
        if numeric_cols and selected_features:
            st.header("📊 Análise Visual das Features")

            # Gráfico de distribuições
            st.subheader("📈 Distribuições das Features Selecionadas")

            # Criar subplots para histogramas
            n_features = len(selected_features)
            cols_per_row = 2
            n_rows = (n_features + cols_per_row - 1) // cols_per_row

            fig_hist = make_subplots(
                rows=n_rows, 
                cols=cols_per_row,
                subplot_titles=selected_features,
                vertical_spacing=0.08
            )

            for i, feature in enumerate(selected_features):
                row = i // cols_per_row + 1
                col = i % cols_per_row + 1

                fig_hist.add_trace(
                    go.Histogram(
                        x=df[feature].dropna(),
                        name=feature,
                        showlegend=False,
                        nbinsx=30
                    ),
                    row=row, col=col
                )

            fig_hist.update_layout(
                height=300 * n_rows,
                title_text="Distribuições das Features Numéricas",
                showlegend=False
            )

            st.plotly_chart(fig_hist, use_container_width=True)

            # Gráfico de série temporal (se aplicável)
            if len(selected_features) > 0:
                st.subheader("📉 Evolução Temporal das Features")

                # Usar índice como proxy para tempo se não houver coluna temporal
                x_axis = df.index if 'timestamp' not in df.columns else df['timestamp']

                fig_time = go.Figure()

                for feature in selected_features[:5]:  # Limitar a 5 features para não sobrecarregar
                    fig_time.add_trace(
                        go.Scatter(
                            x=x_axis,
                            y=df[feature],
                            mode='lines',
                            name=feature,
                            line=dict(width=1)
                        )
                    )

                fig_time.update_layout(
                    title="Evolução das Features ao Longo do Tempo",
                    xaxis_title="Índice/Tempo",
                    yaxis_title="Valor",
                    height=500,
                    hovermode='x unified'
                )

                st.plotly_chart(fig_time, use_container_width=True)

            # Matriz de correlação
            if len(selected_features) > 1:
                st.subheader("🔗 Matriz de Correlação")

                corr_matrix = df[selected_features].corr()

                fig_corr = px.imshow(
                    corr_matrix,
                    title="Matriz de Correlação das Features Selecionadas",
                    color_continuous_scale="RdBu_r",
                    aspect="auto"
                )

                fig_corr.update_layout(height=600)
                st.plotly_chart(fig_corr, use_container_width=True)

        # Seção 3: Download dos Dados
        st.header("💾 Download dos Dados")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Botão para download do CSV completo
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Baixar dados completos (CSV)",
                data=csv_data,
                file_name="features_completos.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            # Botão para download apenas das features selecionadas
            if 'selected_features' in locals() and selected_features:
                selected_csv = df[selected_features].to_csv(index=False)
                st.download_button(
                    label="📥 Baixar features selecionadas (CSV)",
                    data=selected_csv,
                    file_name="features_selecionadas.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with col3:
            # Botão para download das estatísticas
            stats_csv = df.describe().to_csv()
            st.download_button(
                label="📥 Baixar estatísticas (CSV)",
                data=stats_csv,
                file_name="estatisticas_features.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Seção 4: Informações Adicionais
        with st.expander("ℹ️ Informações Técnicas"):
            st.markdown("""
            ### Sobre este Dashboard:
            - **Dados**: Carregados de `data/processed/features.parquet`
            - **Visualizações**: Plotly para gráficos interativos
            - **Performance**: Cache implementado para carregamento eficiente
            - **Exportação**: Múltiplas opções de download em CSV

            ### Features Típicas Esperadas:
            - `log_return_1`: Log return de 1 período
            - `price_zscore_5min`: Z-score do preço em janela de 5 minutos
            - Outras features de engenharia temporal e estatística

            ### Como usar:
            1. Use os filtros na barra lateral para personalizar
            2. Explore as visualizações interativas
            3. Baixe os dados para análises externas
            """)

    else:
        st.error("❌ Não foi possível carregar os dados. Verifique se o arquivo `data/processed/features.parquet` existe.")
        st.info("💡 Para usar este dashboard, certifique-se de que os dados processados estão disponíveis no caminho especificado.")

    # Footer
    st.markdown("---")
    st.markdown("🚀 **High-Frequency Trading Analytics Dashboard** - Desenvolvido com Streamlit")
