document.addEventListener('DOMContentLoaded', () => {
    const analysisForm = document.getElementById('analysis-form');
    const resultsSection = document.getElementById('results-section');
    const engagementMeter = document.getElementById('engagement-meter');
    const engagementValue = document.getElementById('engagement-value');
    const suggestionsList = document.getElementById('suggestions-list');
    
    analysisForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(analysisForm);
        const analyzeBtn = document.getElementById('analyze-btn');
        
        //Cambiar botón a estado de carga
        analyzeBtn.textContent = 'Analizando...';
        analyzeBtn.disabled = true;
        
        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Error en la respuesta del servidor');
            }
            
            const data = await response.json();
            
            //Mostrar resultados
            displayResults(data);
            
            //Restaurar botón
            analyzeBtn.textContent = 'Analizar';
            analyzeBtn.disabled = false;
            
            //Mostrar sección de resultados
            resultsSection.style.display = 'block';
            
            //Desplazar a la sección de resultados
            resultsSection.scrollIntoView({ behavior: 'smooth' });
            
        } catch (error) {
            console.error('Error:', error);
            alert('Ocurrió un error al analizar el contenido. Por favor, intenta nuevamente.');
            
            //Restaurar botón
            analyzeBtn.textContent = 'Analizar';
            analyzeBtn.disabled = false;
        }
    });
    
    function displayResults(data) {
        //Convertir engagement a porcentaje (0-100%)
        const engagementPercentage = Math.min(data.predicted_engagement * 100, 100);
        
        //Actualizar medidor de engagement
        engagementMeter.style.width = `${engagementPercentage}%`;
        engagementValue.textContent = `${engagementPercentage.toFixed(1)}%`;
        
        //Cambiar color según el valor
        if (engagementPercentage < 5) {
            engagementMeter.style.backgroundColor = 'var(--danger-color)';
        } else if (engagementPercentage < 10) {
            engagementMeter.style.backgroundColor = 'var(--warning-color)';
        } else {
            engagementMeter.style.backgroundColor = 'var(--success-color)';
        }
        
        //Mostrar sugerencias
        suggestionsList.innerHTML = '';
        data.suggestions.forEach(suggestion => {
            const li = document.createElement('li');
            li.textContent = suggestion;
            suggestionsList.appendChild(li);
        });
    }
});