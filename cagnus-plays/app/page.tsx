import ChessboardSelector from "./ChessboardSelector";

export default function Home() {
  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh', background: '#181818' }}>
      <div style={{ width: '100vw', maxWidth: 2400, padding: 5 }}>
        <ChessboardSelector />
      </div>
    </div>
  );
}
