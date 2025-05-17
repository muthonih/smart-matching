document.getElementById('matchForm').addEventListener('submit', function (e) {
  e.preventDefault();

  const form = e.target;
  const formData = new FormData(form);
  const data = Object.fromEntries(formData.entries());

  // Optional: Log data for debugging
  console.log("Form Data Submitted:", data);

  fetch('http://127.0.0.1:5000/match', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then(result => {
      const resultDiv = document.getElementById('result');

      if (result.match_success === 1) {
        resultDiv.textContent = '✅ Match Found!';
        resultDiv.style.color = 'green';
      } else if (result.match_success === 0) {
        resultDiv.textContent = `❌ No Match: ${result.reason}`;
        resultDiv.style.color = 'red';
      } else if (result.error) {
        resultDiv.textContent = `⚠️ Error: ${result.error}`;
        resultDiv.style.color = 'orange';
      } else {
        resultDiv.textContent = '❓ Unexpected response from server.';
        resultDiv.style.color = 'gray';
      }
    })
    .catch(error => {
      const resultDiv = document.getElementById('result');
      resultDiv.textContent = '❌ Failed to connect to server.';
      resultDiv.style.color = 'red';
      console.error('Error:', error);
    });
});
