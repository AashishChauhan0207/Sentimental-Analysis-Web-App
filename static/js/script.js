document.addEventListener('DOMContentLoaded', function() {
    const sentimentForm = document.getElementById('sentiment-form');
    const reviewTextarea = document.getElementById('review-text');
    const clearBtn = document.getElementById('clear-btn');
    const resultContainer = document.getElementById('result-container');
    const sentimentIcon = document.getElementById('sentiment-icon');
    const sentimentResult = document.getElementById('sentiment-result');
    
    // Progress bars and scores
    const positiveProgress = document.getElementById('positive-progress');
    const neutralProgress = document.getElementById('neutral-progress');
    const negativeProgress = document.getElementById('negative-progress');
    const positiveScore = document.getElementById('positive-score');
    const neutralScore = document.getElementById('neutral-score');
    const negativeScore = document.getElementById('negative-score');

    // Event listeners
    sentimentForm.addEventListener('submit', analyzeSentiment);
    clearBtn.addEventListener('click', clearForm);

    // Functions
    function analyzeSentiment(e) {
        e.preventDefault();
        
        const reviewText = reviewTextarea.value.trim();
        
        if (!reviewText) {
            alert('Please enter some text to analyze');
            return;
        }
        
        // Show loading state
        const submitBtn = sentimentForm.querySelector('button[type="submit"]');
        const originalBtnText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
        submitBtn.disabled = true;
        
        // Send request to the server
        fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: reviewText })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Reset button state
            submitBtn.innerHTML = originalBtnText;
            submitBtn.disabled = false;
            
            // Display the result
            displayResult(data);
        })
        .catch(error => {
            console.error('Error:', error);
            // Reset button state
            submitBtn.innerHTML = originalBtnText;
            submitBtn.disabled = false;
            
            alert('An error occurred while analyzing. Please try again.');
        });
    }
    
    function displayResult(data) {
        // Set the sentiment text and icon
        const sentiment = data.sentiment;
        let iconClass, resultClass;
        
        // Set appropriate icon and class based on sentiment
        if (sentiment === 'Positive') {
            iconClass = 'fa-face-smile';
            resultClass = 'positive-result';
        } else if (sentiment === 'Neutral') {
            iconClass = 'fa-face-meh';
            resultClass = 'neutral-result';
        } else {
            iconClass = 'fa-face-frown';
            resultClass = 'negative-result';
        }
        
        // Update the result elements
        sentimentIcon.innerHTML = `<i class="fas ${iconClass}"></i>`;
        sentimentResult.textContent = `${sentiment} Sentiment`;
        
        // Remove previous classes and add the new one
        resultContainer.classList.remove('positive-result', 'neutral-result', 'negative-result');
        resultContainer.querySelector('.result-card').classList.remove('positive-result', 'neutral-result', 'negative-result');
        resultContainer.querySelector('.result-card').classList.add(resultClass);
        
        // Update progress bars
        const confidenceScores = data.confidence;
        updateProgressBar(positiveProgress, positiveScore, confidenceScores['Positive'] || 0);
        updateProgressBar(neutralProgress, neutralScore, confidenceScores['Neutral'] || 0);
        updateProgressBar(negativeProgress, negativeScore, confidenceScores['Negative'] || 0);
        
        // Show the result container with animation
        resultContainer.classList.remove('d-none');
        resultContainer.classList.add('result-animation');
        
        // Scroll to result
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    
    function updateProgressBar(progressBar, scoreElement, value) {
        progressBar.style.width = `${value}%`;
        scoreElement.textContent = `${value}%`;
    }
    
    function clearForm() {
        reviewTextarea.value = '';
        resultContainer.classList.add('d-none');
        reviewTextarea.focus();
    }

    // Add some example texts for quick testing
    const exampleTexts = [
        "This product exceeded my expectations! The quality is outstanding and it works perfectly.",
        "Not what I expected. It stopped working after a week and customer service was unhelpful.",
        "It's okay I guess. Does the job but nothing special about it."
    ];
    
    // Add example buttons below textarea
    const exampleContainer = document.createElement('div');
    exampleContainer.className = 'mt-2 example-container';
    exampleContainer.innerHTML = '<small class="text-muted">Try an example:</small>';
    
    const buttonContainer = document.createElement('div');
    buttonContainer.className = 'd-flex flex-wrap gap-2 mt-1';
    
    exampleTexts.forEach((text, index) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'btn btn-sm btn-outline-secondary';
        button.textContent = `Example ${index + 1}`;
        button.addEventListener('click', () => {
            reviewTextarea.value = text;
            resultContainer.classList.add('d-none');
        });
        buttonContainer.appendChild(button);
    });
    
    exampleContainer.appendChild(buttonContainer);
    reviewTextarea.parentNode.appendChild(exampleContainer);
}); 
