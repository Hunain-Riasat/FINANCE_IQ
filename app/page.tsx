export default function Home() {
  return (
    <div className="flex min-h-screen items-center justify-center font-sans">
      <main className="flex w-full max-w-3xl flex-col items-center gap-8 px-6 py-16 text-center sm:items-start sm:text-left">
        <div className="flex flex-col gap-4">
          <h1 className="text-4xl font-bold tracking-tight">
            Personal Finance Manager
          </h1>
          <p className="max-w-md text-lg text-muted-foreground">
            This is a Python Streamlit application. To run it, use: <code className="text-sm bg-slate-100 px-2 py-1 rounded">streamlit run app.py</code>
          </p>
        </div>
      </main>
    </div>
  );
}
