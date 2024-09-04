import React from 'react';
import CategorySelector from './CategorySelector';

function EmailItem({ email, onClassify, onMove }) {
  return (
    <div className="bg-white p-4 rounded shadow">
      <h2 className="font-bold text-lg">{email.subject}</h2>
      <p className="mt-2 text-sm text-gray-600">{email.content}</p>
      <div className="mt-4 flex justify-between items-center">
        <button
          className="bg-blue-500 text-white px-3 py-1 rounded hover:bg-blue-600"
          onClick={() => onClassify(email.id, email.content)}
        >
          Classify
        </button>
        <CategorySelector onSelect={(category) => onMove(email.id, category)} />
      </div>
    </div>
  );
}

export default EmailItem;