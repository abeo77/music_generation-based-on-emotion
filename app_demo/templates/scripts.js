document.addEventListener('DOMContentLoaded', function() {
    // Initialize the app
    initializePage();
});

function initializePage() {
    // Common initialization
    setupTheme();
    animateElements();
    
    // Page specific initialization
    if (document.querySelector('.mood-buttons')) {
        initializeGeneratorPage();
    } else if (document.querySelector('audio')) {
        initializeResultPage();
    }
}

function setupTheme() {
    // Add a class to the body for potential theme switching in future
    document.body.classList.add('light-theme');
}

function animateElements() {
    // Add fade-in animation for key elements
    const elements = document.querySelectorAll('.section, h1, .audio-container');
    
    elements.forEach((el, index) => {
        el.style.animationDelay = (index * 0.1) + 's';
        el.classList.add('animated');
    });
}

// Generator page functions
function initializeGeneratorPage() {
    setupMoodButtons();
    setupSliders();
}

function setupMoodButtons() {
    // Handle mood button selection
    const moodButtons = document.querySelectorAll('.mood-btn');
    
    moodButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove active class from all buttons
            moodButtons.forEach(btn => btn.classList.remove('active-mood'));
            
            // Add active class to clicked button
            this.classList.add('active-mood');
            
            // Set the hidden input value
            document.getElementById('selected_mood').value = this.getAttribute('data-mood');
            
            // Set slider values based on mood
            const mood = this.getAttribute('data-mood');
            
            if(mood === 'happy') {
                setSliderValues(8, 8, 7);
            } else if(mood === 'sad') {
                setSliderValues(2, 3, 2);
            } else if(mood === 'angry') {
                setSliderValues(3, 9, 5);
            } else if(mood === 'relaxed') {
                setSliderValues(5, 2, 2);
            }
        });
    });
}

function setupSliders() {
    // Set up event listeners for all sliders
    const sliders = document.querySelectorAll('input[type="range"]');
    
    sliders.forEach(slider => {
        const valueElement = document.getElementById(slider.id + '_value');
        if (valueElement) {
            updateSliderValue(slider, valueElement);
            
            slider.addEventListener('input', function() {
                updateSliderValue(this, valueElement);
            });
        }
    });
}

function updateSliderValue(slider, valueElement) {
    valueElement.textContent = slider.value;
    
    // Animate the value change
    valueElement.classList.remove('value-change');
    void valueElement.offsetWidth; // Trigger reflow to restart animation
    valueElement.classList.add('value-change');
}

function setSliderValues(valence, energy, danceability) {
    // Set values and update displays for each parameter
    updateSliderWithAnimation('valence', valence);
    updateSliderWithAnimation('energy', energy);
    updateSliderWithAnimation('danceability', danceability);
}

function updateSliderWithAnimation(sliderId, value) {
    const slider = document.getElementById(sliderId);
    const valueElement = document.getElementById(sliderId + '_value');
    
    if (slider && valueElement) {
        // Animate the slider movement
        slider.style.transition = 'all 0.5s ease';
        slider.value = value;
        valueElement.textContent = value;
        
        // Animate the value change
        valueElement.classList.remove('value-change');
        void valueElement.offsetWidth; // Trigger reflow to restart animation
        valueElement.classList.add('value-change');
        
        // Remove transition after animation
        setTimeout(() => {
            slider.style.transition = '';
        }, 500);
    }
}

// Result page functions
function initializeResultPage() {
    setupAudioPlayers();
    setupDownloadButtons();
}

function setupAudioPlayers() {
    const audioPlayers = document.querySelectorAll('audio');
    
    audioPlayers.forEach(player => {
        // Add event listeners for audio controls
        player.addEventListener('play', function() {
            // Highlight the currently playing container
            const container = this.closest('.audio-container');
            if (container) {
                container.classList.add('playing');
            }
        });
        
        player.addEventListener('pause', function() {
            // Remove highlight from container
            const container = this.closest('.audio-container');
            if (container) {
                container.classList.remove('playing');
            }
        });
        
        player.addEventListener('ended', function() {
            // Remove highlight from container
            const container = this.closest('.audio-container');
            if (container) {
                container.classList.remove('playing');
            }
        });
    });
}

function setupDownloadButtons() {
    const downloadButtons = document.querySelectorAll('a[download]');
    
    downloadButtons.forEach(button => {
        // Add animation for download buttons
        button.addEventListener('click', function() {
            this.classList.add('pulse');
            
            setTimeout(() => {
                this.classList.remove('pulse');
            }, 600);
        });
    });
} 