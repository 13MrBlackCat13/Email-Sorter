import React, { useState, useEffect } from 'react';
import EmailList from './components/EmailList';
import axios from 'axios';

function App() {
  const [emails, setEmails] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchEmails();
  }, []);

  const fetchEmails = async () => {
    try {
      const response = await axios.get('/api/emails');
      setEmails(response.data);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching emails:', error);
      setLoading(false);
    }
  };

  const handleClassify = async (emailId, content) => {
    try {
      const response = await axios.post('/api/classify', { content });
      alert(`Classification results:\n${JSON.stringify(response.data, null, 2)}`);
    } catch (error) {
      console.error('Error classifying email:', error);
    }
  };

  const handleMove = async (emailId, category) => {
    try {
      await axios.post('/api/move', { id: emailId, category });
      setEmails(emails.filter(email => email.id !== emailId));
    } catch (error) {
      console.error('Error moving email:', error);
    }
  };

  const handleRetrain = async () => {
    try {
      const response = await axios.post('/api/retrain');
      alert(response.data.message);
    } catch (error) {
      console.error('Error retraining model:', error);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-4">Email Sorter</h1>
      {loading ? (
        <p>Loading emails...</p>
      ) : (
        <>
          <EmailList
            emails={emails}
            onClassify={handleClassify}
            onMove={handleMove}
          />
          <button
            className="mt-4 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
            onClick={handleRetrain}
          >
            Retrain Model
          </button>
        </>
      )}
    </div>
  );
}

export default App;