async function sendMessage() {
    const input = document.getElementById("user-input");
    const message = input.value.trim();

    if (message === "") return;

    // Display user message
    const messagesDiv = document.getElementById("messages");
    const userMessage = document.createElement("div");
    userMessage.textContent = `You: ${message}`;
    userMessage.style.marginBottom = "10px";
    messagesDiv.appendChild(userMessage);

    // Clear input field
    input.value = "";

    // Send message to the server
    const response = await fetch("/generative_response", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ message })
    });

    const data = await response.json();

    // Display bot response
    const botMessage = document.createElement("div");
    botMessage.textContent = `Bot: ${data.response}`;
    botMessage.style.marginBottom = "10px";
    botMessage.style.color = "blue";
    messagesDiv.appendChild(botMessage);

    // Scroll to the bottom
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}
