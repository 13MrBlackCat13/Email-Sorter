document.addEventListener('DOMContentLoaded', function() {
    fetchEmails();
});

function fetchEmails() {
    fetch('/get_emails')
        .then(response => response.json())
        .then(emails => {
            const emailList = document.getElementById('emailList');
            emailList.innerHTML = '';
            emails.forEach(email => {
                const emailElement = createEmailElement(email);
                emailList.appendChild(emailElement);
            });
        })
        .catch(error => console.error('Error:', error));
}

function createEmailElement(email) {
    const div = document.createElement('div');
    div.className = 'bg-white p-4 rounded shadow';
    div.innerHTML = `
        <h2 class="font-bold">${email.filename}</h2>
        <p class="text-sm text-gray-600">Category: ${email.category}</p>
        <p class="mt-2">${email.content}</p>
        <button onclick="classifyEmail('${email.filename}', '${email.content}')" class="mt-2 bg-blue-500 text-white px-2 py-1 rounded">Classify</button>
        <select onchange="moveEmail('${email.filename}', '${email.category}', this.value)" class="mt-2 border rounded">
            <option value="">Move to...</option>
            <option value="Входящие">Входящие</option>
            <option value="Рассылки">Рассылки</option>
            <option value="Социальные сети">Социальные сети</option>
            <option value="Чеки_Квитанции">Чеки/Квитанции</option>
            <option value="Новости">Новости</option>
            <option value="Доставка">Доставка</option>
            <option value="Госписьма">Госписьма</option>
            <option value="Учёба">Учёба</option>
            <option value="Игры">Игры</option>
            <option value="Spam/Мошенничество">Spam/Мошенничество</option>
            <option value="Spam/Обычный">Spam/Обычный</option>
        </select>
    `;
    return div;
}

function classifyEmail(filename, content) {
    fetch('/classify_email', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content: content }),
    })
    .then(response => response.json())
    .then(data => {
        alert(`Classification results:\n${JSON.stringify(data, null, 2)}`);
    })
    .catch(error => console.error('Error:', error));
}

function moveEmail(filename, fromCategory, toCategory) {
    if (!toCategory) return;

    fetch('/move_email', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ filename: filename, from_category: fromCategory, to_category: toCategory }),
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        fetchEmails();  // Refresh the email list
    })
    .catch(error => console.error('Error:', error));
}