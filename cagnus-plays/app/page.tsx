import ChessboardSelector from "./ChessboardSelector";

export default function Home() {
  return (
    <div style={{ minHeight: '100vh', background: '#fff', fontFamily: 'Inter, Arial, sans-serif', position: 'relative' }}>
      {/* Norwegian flag in top right */}
      <a href="https://en.wikipedia.org/wiki/Magnus_Carlsen" target="_blank" rel="noopener noreferrer">
        <img
          src="https://upload.wikimedia.org/wikipedia/commons/d/d9/Flag_of_Norway.svg"
          alt="Norwegian Flag"
          style={{ position: 'fixed', top: 24, right: 32, width: 300, height: 200, boxShadow: '0 2px 12px rgba(0,0,0,0.18)', borderRadius: 2, border: '3px solid #fff', zIndex: 10 }}
        />
      </a>
      {/* Main content area */}
      <div
        style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-start',
          width: '100vw', minHeight: '100vh',
          paddingTop: 48, paddingBottom: 48,
        }}
      >
        {/* Title */}
        <h1
          style={{
            fontSize: '2.8rem',
            fontWeight: 900,
            letterSpacing: '0.04em',
            marginBottom: 32,
            textAlign: 'center',
            fontFamily: 'Inter, Arial, sans-serif',
            WebkitBackgroundClip: 'text',
            textShadow: '0 2px 8px #00286822, 0 1px 0 #fff',
          }}
        >
          Cagnus Marlsen Chess Championships
        </h1>
        {/* Video area with Norwegian flag colors */}
        <div
          style={{
            background: '#181c23',
            borderRadius: 5,
            position: 'relative',
            padding: 2,
            width: '100vw',
            maxWidth: 1600,
            margin: '0 auto',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            transition: 'box-shadow 0.2s',
            boxSizing: 'border-box',
          }}
        >
          <div
            style={{
              borderRadius: 5,
              padding: 0,
              background: '#181c23',
              width: '100%',
              boxSizing: 'border-box',
              display: 'flex',
              justifyContent: 'center',
            }}
          >
            <ChessboardSelector />
          </div>
        </div>
      </div>
    </div>
  );
}
