"use client";
import { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function BaumWelchMenu() {
  const [iterations, setIterations] = useState(100);
  const [states, setStates] = useState(2);
  const [sequence, setSequence] = useState("1, 0, 1, 1, 0");

  const [matrixA, setMatrixA] = useState("[[0.7, 0.3], [0.4, 0.6]]");
  const [matrixB, setMatrixB] = useState("[[0.9, 0.1], [0.2, 0.8]]");
  const [pi, setPi] = useState("[0.6, 0.4]");

  // NEW UI STATES
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [loading, setLoading] = useState(false);

  const observationCount = sequence
    .split(',')
    .map(item => item.trim())
    .filter(item => item !== "").length;

  const handleRunAlgorithm = async () => {
    setErrorMsg("");
    setResult(null);
    setLoading(true);

    let parsedA, parsedB, parsedPi;
    try {
      parsedA = JSON.parse(matrixA);
      parsedB = JSON.parse(matrixB);
      parsedPi = JSON.parse(pi);
    } catch (error) {
      setErrorMsg("Invalid JSON format in one of your matrices. Please check your brackets!");
      setLoading(false);
      return;
    }

    try {
      const payload = {
        iterations: Number(iterations),
        states: Number(states),
        sequence: sequence.split(',').map(item => parseInt(item.trim())).filter(n => !isNaN(n)),
        matrix_a: parsedA,
        matrix_b: parsedB,
        pi: parsedPi
      };

      const response = await fetch("http://127.0.0.1:8000/api/run/ ", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
         const errorText = await response.text();
         console.error("Django Error:", errorText);
         setErrorMsg(`Server crashed (Status ${response.status}). Check Django terminal for the math error.`);
         setLoading(false);
         return;
      }

      const data = await response.json();
      setResult(data);
      
    } catch (error) {
      setErrorMsg("Network error. Is your Django server running?");
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-50 pt-10 px-4 pb-20">
      <h1 className="text-3xl font-light text-gray-800 text-center tracking-wide mb-8">
        Baum-Welch Implementation
      </h1>

      <div className="max-w-5xl mx-auto space-y-8">
        
        {/* INPUT SECTION */}
        <div className="bg-white shadow-sm border border-gray-100 rounded-xl p-6 md:p-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <div className="flex flex-col">
              <label className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wider">Iterations</label>
              <input type="number" min="1" value={iterations} onChange={e => setIterations(e.target.value)} className="p-3 rounded-lg border border-gray-200 bg-gray-50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-gray-300 transition-all" />
            </div>
            <div className="flex flex-col">
              <label className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wider">Hidden States</label>
              <input type="number" min="2" value={states} onChange={e => setStates(e.target.value)} className="p-3 rounded-lg border border-gray-200 bg-gray-50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-gray-300 transition-all" />
            </div>
            <div className="flex flex-col">
              <div className="flex justify-between items-end mb-2">
                <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Sequence</label>
                <span className={`text-xs font-bold px-2 py-0.5 rounded ${observationCount < 2 ? 'bg-red-100 text-red-600' : 'bg-green-100 text-green-700'}`}>Count: {observationCount}</span>
              </div>
              <input type="text" value={sequence} onChange={e => setSequence(e.target.value)} className="p-3 rounded-lg border border-gray-200 bg-gray-50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-gray-300 transition-all font-mono text-sm" />
            </div>
          </div>

          <div className="space-y-6 mb-8">
            <div className="flex flex-col">
              <label className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wider">Transition Matrix A</label>
              <textarea rows="2" value={matrixA} onChange={e => setMatrixA(e.target.value)} className="w-full p-3 rounded-lg border border-gray-200 focus:bg-white focus:outline-none focus:ring-2 focus:ring-gray-300 transition-all font-mono text-sm resize-y" />
            </div>
            <div className="flex flex-col">
              <label className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wider">Emission Matrix B</label>
              <textarea rows="2" value={matrixB} onChange={e => setMatrixB(e.target.value)} className="w-full p-3 rounded-lg border border-gray-200 focus:bg-white focus:outline-none focus:ring-2 focus:ring-gray-300 transition-all font-mono text-sm resize-y" />
            </div>
            <div className="flex flex-col">
              <label className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wider">Initial Probabilities π</label>
              <input type="text" value={pi} onChange={e => setPi(e.target.value)} className="w-full p-3 rounded-lg border border-gray-200 focus:bg-white focus:outline-none focus:ring-2 focus:ring-gray-300 transition-all font-mono text-sm" />
            </div>
          </div>

          {errorMsg && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm font-medium">
              ⚠️ {errorMsg}
            </div>
          )}

          <div className="flex justify-end">
            <button 
              onClick={handleRunAlgorithm}
              disabled={observationCount < 2 || loading}
              className="w-full md:w-auto bg-gray-800 hover:bg-gray-900 disabled:bg-gray-300 text-white font-medium py-3 px-8 rounded-lg transition-colors"
            >
              {loading ? "Calculating..." : "Run Algorithm"}
            </button>
          </div>
        </div>

        {/* RESULTS SECTION */}
        {result && (
          <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            
            {/* 1. Final Value Div */}
            <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-8 text-center shadow-sm">
              <h2 className="text-emerald-800 font-semibold uppercase tracking-wider text-sm mb-2">Final Sequence Probability</h2>
              <div className="text-5xl font-light text-emerald-900">
                {result.final_probability.toExponential(4)}
              </div>
            </div>


            {/* 3. Graph of Iterations */}
            <div className="bg-white shadow-sm border border-gray-100 rounded-xl p-6">
              <h3 className="text-lg font-medium text-gray-800 mb-6">Probability Convergence</h3>
              <div className="h-72 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={result.history}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eee" />
                    <XAxis dataKey="iteration" tick={{fontSize: 12}} tickMargin={10} />
                    <YAxis 
                      domain={['auto', 'auto']} 
                      tickFormatter={(val) => val.toExponential(2)} 
                      tick={{fontSize: 12}} 
                      width={80}
                    />
                    <Tooltip 
                      formatter={(value) => value.toExponential(4)}
                      labelFormatter={(label) => `Iteration ${label}`}
                    />
                    <Line type="monotone" dataKey="probability" stroke="#1f2937" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* 4. Table of Iterations */}
            <div className="bg-white shadow-sm border border-gray-100 rounded-xl overflow-hidden">
              <div className="px-6 py-4 border-b border-gray-100">
                <h3 className="text-lg font-medium text-gray-800">Iteration History</h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm text-left text-gray-500">
                  <thead className="text-xs text-gray-700 uppercase bg-gray-50">
                    <tr>
                      <th className="px-6 py-4">Iter</th>
                      <th className="px-6 py-4">Probability</th>
                      <th className="px-6 py-4">Initial (π)</th>
                      <th className="px-6 py-4">Transition (A)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.history.map((step) => (
                      <tr key={step.iteration} className="border-b hover:bg-gray-50/50">
                        <td className="px-6 py-4 font-medium text-gray-900">{step.iteration}</td>
                        <td className="px-6 py-4 font-mono">{step.probability.toExponential(4)}</td>
                        <td className="px-6 py-4 font-mono text-xs">
                          {JSON.stringify(step.pi.map(n => Number(n.toFixed(3))))}
                        </td>
                        <td className="px-6 py-4 font-mono text-xs">
                          {JSON.stringify(step.matrix_a.map(row => row.map(n => Number(n.toFixed(3)))))}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

          </div>
        )}
      </div>
    </div>
  );
}
