const express = require('express');
const path = require('path');
const app = express();

app.use(express.static(__dirname)); // Serve all files in current folder

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

const PORT = 3000;
app.listen(PORT, () => {
  console.log(`Frontend running at http://localhost:${PORT}`);
});
