import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import ollama

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


def safe_series(df, column_name):
    """Devuelve una serie numérica segura o una serie vacía si la columna no existe."""
    if column_name not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column_name], errors='coerce')


def safe_mean(df, column_name):
    series = safe_series(df, column_name)
    return series.mean() if not series.empty else 0.0


def outcome_rate(df, outcome_value):
    if 'Outcome/Type' not in df.columns or len(df) == 0:
        return 0.0
    return (pd.to_numeric(df['Outcome/Type'], errors='coerce') == outcome_value).mean() * 100


def split_mean_delta(first_half, second_half, column_name):
    first = safe_mean(first_half, column_name)
    second = safe_mean(second_half, column_name)
    return first, second, second - first


def data_coverage(df, columns):
    coverage = {}
    for col in columns:
        if col in df.columns and len(df) > 0:
            valid_ratio = pd.to_numeric(df[col], errors='coerce').notna().mean() * 100
            coverage[col] = valid_ratio
        else:
            coverage[col] = 0.0
    return coverage

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
        
        # Mapeo real de Outcome/Type en entrenamiento:
        # 0 = timeout, 1 = collision, 3 = out_of_road, 4 = success
        success_rate = outcome_rate(df, 4.0)
        collision_rate = outcome_rate(df, 1.0)
        timeout_rate = outcome_rate(df, 0.0)
        
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
            
        # 2. Reducción de Colisiones (outcome == 1)
        col_first = outcome_rate(first_half, 1.0)
        col_second = outcome_rate(second_half, 1.0)
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
        
        st.subheader("Analista IA")
        
        # Seleccionador de modelo
        modelo_seleccionado = 'gemma3:12b'
        
        if st.button("Generar Insights de Entrenamiento"):
            with st.spinner(f"Analizando datos con {modelo_seleccionado}... esto puede tardar unos segundos."):
                
                # 1. Preparar el resumen estadístico robusto para el LLM
                half_point = len(df) // 2
                first_half = df.iloc[:half_point]
                second_half = df.iloc[half_point:]
                
                reward_1, reward_2, reward_delta = split_mean_delta(first_half, second_half, 'Reward/Raw_Episode')
                kl_1, kl_2, kl_delta = split_mean_delta(first_half, second_half, 'Training/Approx_KL')
                entropy_1, entropy_2, entropy_delta = split_mean_delta(first_half, second_half, 'Training/Entropy')
                episode_len_1, episode_len_2, episode_len_delta = split_mean_delta(first_half, second_half, 'Training/Episode_Length')

                shield_1, shield_2, shield_delta = split_mean_delta(first_half, second_half, 'Safety/Shield_Activations')
                lane_1, lane_2, lane_delta = split_mean_delta(first_half, second_half, 'CARLA/Lane_Invasions_Ep')
                carla_col_1, carla_col_2, carla_col_delta = split_mean_delta(first_half, second_half, 'CARLA/Collisions_Ep')
                lateral_1, lateral_2, lateral_delta = split_mean_delta(first_half, second_half, 'CARLA/Lateral_Offset_Norm')
                speed_1, speed_2, speed_delta = split_mean_delta(first_half, second_half, 'CARLA/Speed_kmh')
                dist_1, dist_2, dist_delta = split_mean_delta(first_half, second_half, 'CARLA/Total_Distance')
                policy_loss_1, policy_loss_2, policy_loss_delta = split_mean_delta(first_half, second_half, 'Loss/Policy_Loss')
                value_loss_1, value_loss_2, value_loss_delta = split_mean_delta(first_half, second_half, 'Loss/Value_Loss')

                # Outcome/Type real: 0 timeout, 1 collision, 3 out_of_road, 4 success
                success_1 = outcome_rate(first_half, 4.0)
                success_2 = outcome_rate(second_half, 4.0)
                collision_1 = outcome_rate(first_half, 1.0)
                collision_2 = outcome_rate(second_half, 1.0)
                road_1 = outcome_rate(first_half, 3.0)
                road_2 = outcome_rate(second_half, 3.0)
                timeout_1 = outcome_rate(first_half, 0.0)
                timeout_2 = outcome_rate(second_half, 0.0)

                quality_cols = [
                    'Training/Approx_KL',
                    'Training/Entropy',
                    'Loss/Policy_Loss',
                    'Loss/Value_Loss',
                    'Safety/Shield_Activations',
                    'CARLA/Lane_Invasions_Ep',
                    'CARLA/Collisions_Ep',
                    'CARLA/Lateral_Offset_Norm',
                ]
                coverage = data_coverage(df, quality_cols)
                high_kl_ratio = (safe_series(df, 'Training/Approx_KL') > 0.05).mean() * 100 if len(df) > 0 else 0.0
                
                # Construimos un prompt técnico orientado a seguridad para CARLA
                prompt_estadistico = f"""
                Eres un evaluador técnico de entrenamiento RL para conducción autónoma en CARLA con prioridad absoluta de seguridad.

                CONTEXTO DEL EXPERIMENTO:
                - Entorno: conducción autónoma en CARLA.
                - Episodios totales: {len(df)}.
                - Convención Outcome/Type validada: 0=timeout, 1=collision, 3=out_of_road, 4=success.

                CALIDAD DE DATOS (cobertura no-NaN):
                - KL: {coverage['Training/Approx_KL']:.1f}% | Entropía: {coverage['Training/Entropy']:.1f}%
                - Policy Loss: {coverage['Loss/Policy_Loss']:.1f}% | Value Loss: {coverage['Loss/Value_Loss']:.1f}%
                - Shield Activations: {coverage['Safety/Shield_Activations']:.1f}% | Lane Invasions: {coverage['CARLA/Lane_Invasions_Ep']:.1f}%
                - Collisions/Ep: {coverage['CARLA/Collisions_Ep']:.1f}% | Lateral Offset: {coverage['CARLA/Lateral_Offset_Norm']:.1f}%

                COMPARATIVA PRIMERA MITAD vs SEGUNDA MITAD:
                - Reward medio: {reward_1:.2f} -> {reward_2:.2f} (delta {reward_delta:+.2f})
                - Success rate (outcome=4): {success_1:.1f}% -> {success_2:.1f}%
                - Collision rate (outcome=1): {collision_1:.1f}% -> {collision_2:.1f}%
                - Out-of-road rate (outcome=3): {road_1:.1f}% -> {road_2:.1f}%
                - Timeout rate (outcome=0): {timeout_1:.1f}% -> {timeout_2:.1f}%
                - Shield activations/ep: {shield_1:.2f} -> {shield_2:.2f} (delta {shield_delta:+.2f})
                - Lane invasions/ep: {lane_1:.2f} -> {lane_2:.2f} (delta {lane_delta:+.2f})
                - CARLA collisions/ep: {carla_col_1:.2f} -> {carla_col_2:.2f} (delta {carla_col_delta:+.2f})
                - Lateral offset norm: {lateral_1:.3f} -> {lateral_2:.3f} (delta {lateral_delta:+.3f})
                - Velocidad media (km/h): {speed_1:.2f} -> {speed_2:.2f} (delta {speed_delta:+.2f})
                - Distancia media (m): {dist_1:.2f} -> {dist_2:.2f} (delta {dist_delta:+.2f})
                - Longitud media episodio: {episode_len_1:.1f} -> {episode_len_2:.1f} (delta {episode_len_delta:+.1f})
                - KL media: {kl_1:.4f} -> {kl_2:.4f} (delta {kl_delta:+.4f})
                - % episodios con KL > 0.05: {high_kl_ratio:.1f}%
                - Entropía media: {entropy_1:.4f} -> {entropy_2:.4f} (delta {entropy_delta:+.4f})
                - Policy loss media: {policy_loss_1:.4f} -> {policy_loss_2:.4f} (delta {policy_loss_delta:+.4f})
                - Value loss media: {value_loss_1:.4f} -> {value_loss_2:.4f} (delta {value_loss_delta:+.4f})

                INSTRUCCIONES OBLIGATORIAS:
                1. Prioriza seguridad: primero analiza collision/out-of-road/timeout y estabilidad de control.
                2. No inventes datos ni causas no soportadas por las métricas.
                3. Si hay baja cobertura de datos (<70%), menciona el impacto en la fiabilidad del diagnóstico.
                4. Interpreta KL > 0.05 como señal de inestabilidad de política PPO.
                5. Evalúa si el shield está ayudando (más seguridad) o enmascarando una política deficiente.

                FORMATO DE RESPUESTA (Markdown):
                - Diagnóstico de seguridad (3-5 bullets)
                - Diagnóstico de estabilidad de entrenamiento PPO (2-4 bullets)
                - Riesgo global del run: Bajo/Medio/Alto + justificación numérica breve
                - Top 3 ajustes priorizados para la próxima corrida (cada ajuste con impacto esperado y trade-off)
                """
                
                try:
                    # 2. Llamar a la API local de Ollama
                    response = ollama.chat(model=modelo_seleccionado, messages=[
                        {
                            'role': 'system',
                            'content': 'Eres un asistente experto en IA y Reinforcement Learning que da respuestas precisas y técnicas.'
                        },
                        {
                            'role': 'user', 
                            'content': prompt_estadistico
                        }
                    ])
                    
                    # 3. Mostrar el resultado
                    st.info("**Conclusiones de la IA:**")
                    st.markdown(response['message']['content'])
                    
                except Exception as e:
                    st.error(f"Error al conectar con Ollama. ¿Te aseguraste de ejecutar 'ollama serve' o tener la app abierta? Detalle del error: {e}")