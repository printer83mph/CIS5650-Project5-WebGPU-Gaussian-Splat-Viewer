const html_log = document.querySelector('#log') as HTMLDivElement;

export async function log(msg: string) {
  console.log(msg);
  const p = document.createElement('p');
  p.innerText = msg;
  html_log.appendChild(p);
}

let t: number;

export function time() {
  t = performance.now();
}

export function timeLog() {
  const d = performance.now() - t;
  log(`${d.toFixed(0)} ms`);
}

export function timeReturn() {
  const d = performance.now() - t;
  return d;
}
