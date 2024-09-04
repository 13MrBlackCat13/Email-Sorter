import React from 'react';

const categories = [
  'Входящие', 'Рассылки', 'Социальные сети', 'Чеки_Квитанции',
  'Новости', 'Доставка', 'Госписьма', 'Учёба', 'Игры',
  'Spam/Мошенничество', 'Spam/Обычный'
];

function CategorySelector({ onSelect }) {
  return (
    <select
      className="border rounded px-2 py-1"
      onChange={(e) => onSelect(e.target.value)}
      defaultValue=""
    >
      <option value="" disabled>Move to...</option>
      {categories.map(category => (
        <option key={category} value={category}>{category}</option>
      ))}
    </select>
  );
}

export default CategorySelector;