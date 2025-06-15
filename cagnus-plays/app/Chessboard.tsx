import React from 'react';

const pieceUnicode: Record<string, string> = {
  K: '♔', Q: '♕', R: '♖', B: '♗', N: '♘', P: '♙',
  k: '♚', q: '♛', r: '♜', b: '♝', n: '♞', p: '♟',
  '.': ''
};

function fenToBoard(fen: string): string[][] {
  const rows = fen.split(' ')[0].split('/');
  return rows.map(row => {
    const arr: string[] = [];
    for (const char of row) {
      if (/[1-8]/.test(char)) {
        for (let i = 0; i < parseInt(char); i++) arr.push('.');
      } else {
        arr.push(char);
      }
    }
    return arr;
  });
}

export default function Chessboard({ fen }: { fen?: string }) {
  if (!fen) return <div style={{ color: '#888', fontSize: 14 }}>No board state</div>;
  const board = fenToBoard(fen);
  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(8, 28px)',
      gridTemplateRows: 'repeat(8, 28px)',
      border: '2px solid #e5e7eb',
      borderRadius: 8,
      background: '#f3f4f6',
      boxShadow: '0 1px 4px rgba(0,0,0,0.04)',
      width: 224,
      height: 224,
      margin: '0 auto',
      marginBottom: 8,
    }}>
      {board.map((row, i) =>
        row.map((piece, j) => {
          const isLight = (i + j) % 2 === 0;
          return (
            <div key={j}
              style={{
                width: 28, height: 28,
                background: isLight ? '#f9fafb' : '#d1d5db',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 20, fontWeight: 600,
                color: /[A-Z]/.test(piece) ? '#222' : '#444',
                userSelect: 'none',
              }}>
              {pieceUnicode[piece] || ''}
            </div>
          );
        })
      )}
    </div>
  );
}
