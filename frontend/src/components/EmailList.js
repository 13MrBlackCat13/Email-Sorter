import React from 'react';
import EmailItem from './EmailItem';

function EmailList({ emails, onClassify, onMove }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {emails.map(email => (
        <EmailItem
          key={email.id}
          email={email}
          onClassify={onClassify}
          onMove={onMove}
        />
      ))}
    </div>
  );
}

export default EmailList;