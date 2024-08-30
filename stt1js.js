document.addEventListener('DOMContentLoaded', () => {
    const startButton = document.getElementById('startButton');
    const speechText = document.getElementById('speechText');

    // Check if the browser supports the Web Speech API
    if ('webkitSpeechRecognition' in window) {
        const recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onstart = () => {
            startButton.innerText = 'Listening...';
            startButton.disabled = true;
        };

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            speechText.value = transcript;
        };

        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            startButton.innerText = 'Start Speaking';
            startButton.disabled = false;
        };

        recognition.onend = () => {
            startButton.innerText = 'Start Speaking';
            startButton.disabled = false;
        };

        startButton.addEventListener('click', () => {
            recognition.start();
        });
    } else {
        startButton.innerText = 'Speech recognition not supported';
        startButton.disabled = true;
    }
});
