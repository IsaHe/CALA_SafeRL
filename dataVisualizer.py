import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Configuración de la página
st.set_page_config(page_title="RL Training Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Estilos CSS personalizados para simular el aspecto del HTML
st.markdown("""
    <style>
    .metric-card {
        background-color: #1E1E2E;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #333;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-label { font-size: 12px; color: #A0A0B0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
    .metric-value { font-size: 24px; font-weight: bold; color: #FFFFFF; }
    .insight-card {
        padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid;
        background-color: #1E1E2E;
    }
    .insight-good { border-left-color: #3ecf8e; }
    .insight-warn { border-left-color: #ff4f4f; }
    .insight-info { border-left-color: #4a9eff; }
    </style>
""", unsafe_allow_html=True)

st.title("Dashboard de Entrenamiento RL - Agente Individual")
st.markdown("Analiza la estabilidad, el rendimiento y el comportamiento de tu agente en CARLA.")

# 1. Subida de archivo
uploaded_file = st.file_uploader("Sube el archivo CSV del agente", type=["csv"])

# Función auxiliar para crear gráficas con media móvil
def plot_metric(df, x_col, y_col, title, rolling_window=30, color="#4a9eff", invert_good=False):
    fig = go.Figure()
    
    # Línea cruda (transparente)
    fig.add_trace(go.Scatter(
        x=df[x_col], y=df[y_col], 
        mode='lines', line=dict(color=color, width=1), opacity=0.3,
        name='Crudo'
    ))
    
    # Media móvil
    if rolling_window > 0:
        rolling_series = df[y_col].rolling(window=rolling_window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df[x_col], y=rolling_series, 
            mode='lines', line=dict(color=color, width=2.5),
            name=f'Media ({rolling_window} ep)'
        ))

    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=40, b=20),
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor='#333'),
        yaxis=dict(showgrid=True, gridcolor='#333')
    )
    return fig

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    
    if 'Step' not in df.columns:
        st.error("El CSV no contiene la columna 'Step'.")
    else:
        st.markdown("---")
        
        # --- SECCIÓN 1: MÉTRICAS GLOBALES ---
        st.subheader("Resumen Global")
        
        # Cálculos de métricas
        total_episodes = df['Step'].max()
        avg_reward = df['Reward/Raw_Episode'].mean()
        max_reward = df['Reward/Raw_Episode'].max()
        
        # Mapeo de Outcomes (Asumiendo: 0:Otro/MaxSteps, 1:Éxito, 3:Colisión - Ajustar según tu lógica)
        outcomes = df['Outcome/Type'].value_counts()
        success_rate = (outcomes.get(1.0, 0) / len(df)) * 100
        collision_rate = (outcomes.get(3.0, 0) / len(df)) * 100
        timeout_rate = (outcomes.get(2.0, 0) / len(df)) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Episodios Totales</div><div class="metric-value">{total_episodes:.0f}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Reward Medio</div><div class="metric-value">{avg_reward:.2f}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Tasa de Éxito</div><div class="metric-value" style="color: #3ecf8e;">{success_rate:.1f}%</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Tasa de Colisión</div><div class="metric-value" style="color: #ff4f4f;">{collision_rate:.1f}%</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- SECCIÓN 2: GRÁFICAS DE RENDIMIENTO ---
        st.subheader("Evolución del Entrenamiento")
        
        # Reward
        if 'Reward/Raw_Episode' in df.columns:
            st.plotly_chart(plot_metric(df, 'Step', 'Reward/Raw_Episode', "Reward por Episodio", color="#3ecf8e"), width='stretch')
        
        # Gráficas en 2 columnas
        colA, colB = st.columns(2)
        
        with colA:
            if 'Training/Approx_KL' in df.columns:
                st.plotly_chart(plot_metric(df, 'Step', 'Training/Approx_KL', "KL Divergence (Estabilidad)", color="#f5a623"), width='stretch')
            
            if 'Safety/Shield_Activations' in df.columns:
                st.plotly_chart(plot_metric(df, 'Step', 'Safety/Shield_Activations', "Activaciones del Escudo / Ep", color="#ff4f4f"), width='stretch')
                
            if 'CARLA/Lane_Invasions_Ep' in df.columns:
                st.plotly_chart(plot_metric(df, 'Step', 'CARLA/Lane_Invasions_Ep', "Invasiones de Carril / Ep", color="#a855f7"), width='stretch')

        with colB:
            if 'Training/Entropy' in df.columns:
                st.plotly_chart(plot_metric(df, 'Step', 'Training/Entropy', "Entropía de la Política (Exploración)", color="#4a9eff"), width='stretch')
                
            if 'Training/Episode_Length' in df.columns:
                st.plotly_chart(plot_metric(df, 'Step', 'Training/Episode_Length', "Longitud del Episodio (Steps)", color="#4a9eff"), width='stretch')
                
            if 'Loss/Policy_Loss' in df.columns and 'Loss/Value_Loss' in df.columns:
                # Gráfica combinada de Losses
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(x=df['Step'], y=df['Loss/Policy_Loss'].rolling(20, min_periods=1).mean(), name='Policy Loss', line=dict(color='#4a9eff')))
                fig_loss.add_trace(go.Scatter(x=df['Step'], y=df['Loss/Value_Loss'].rolling(20, min_periods=1).mean(), name='Value Loss', line=dict(color='#f5a623')))
                fig_loss.update_layout(title="Losses (Media 20 ep)", margin=dict(l=20, r=20, t=40, b=20), height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig_loss, width='stretch')

        st.markdown("---")

        # --- SECCIÓN 3: INSIGHTS AUTOMÁTICOS ---
        st.subheader("Insights Automáticos")
        
        half_point = len(df) // 2
        first_half = df.iloc[:half_point]
        second_half = df.iloc[half_point:]
        
        insights = []
        
        # 1. Tendencia del Reward
        reward_diff = second_half['Reward/Raw_Episode'].mean() - first_half['Reward/Raw_Episode'].mean()
        if reward_diff > 5:
            insights.append({"type": "insight-good", "title": "Mejora del Reward", "desc": f"El reward medio ha aumentado en {reward_diff:.1f} puntos en la segunda mitad del entrenamiento. El agente está aprendiendo."})
        elif reward_diff < -5:
            insights.append({"type": "insight-warn", "title": "Caída del Reward", "desc": f"El reward medio ha caído en {abs(reward_diff):.1f} puntos. Revisa si hay olvido catastrófico (catastrophic forgetting)."})
            
        # 2. Reducción de Colisiones
        col_first = (first_half['Outcome/Type'] == 3.0).mean() * 100
        col_second = (second_half['Outcome/Type'] == 3.0).mean() * 100
        if col_first - col_second > 2:
            insights.append({"type": "insight-good", "title": "Conducción más segura", "desc": f"Las colisiones se han reducido del {col_first:.1f}% al {col_second:.1f}%."})
            
        # 3. Estabilidad (KL)
        if 'Training/Approx_KL' in df.columns:
            high_kl = (df['Training/Approx_KL'] > 0.05).sum()
            if high_kl > len(df) * 0.1:
                insights.append({"type": "insight-warn", "title": "Inestabilidad en la Política", "desc": "La divergencia KL ha sido alta en muchos episodios. Considera reducir el Learning Rate o ajustar los hiperparámetros de PPO."})

        # Renderizar Insights
        if not insights:
            st.info("No se han detectado tendencias críticas suficientes para generar insights automáticos.")
        else:
            for insight in insights:
                st.markdown(f"""
                <div class="insight-card {insight['type']}">
                    <strong>{insight['title']}</strong><br>
                    <span style="color: #A0A0B0; font-size: 14px;">{insight['desc']}</span>
                </div>
                """, unsafe_allow_html=True)