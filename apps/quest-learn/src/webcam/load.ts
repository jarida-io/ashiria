/** Imports a module the dev server serves as a plain file. A failed attempt (the server restarting, a dropped
 * connection) is remembered by the browser for the whole page, so the retry asks for it under a fresh URL. */
export async function importFresh(url: string, tries = 3): Promise<any> {
  for (let k = 0; ; k++) {
    try {
      return await import(/* @vite-ignore */ k ? `${url}?retry=${Date.now()}` : url);
    } catch (err) {
      if (k + 1 >= tries) throw err;
      await new Promise((r) => setTimeout(r, 1000 * (k + 1)));
    }
  }
}
