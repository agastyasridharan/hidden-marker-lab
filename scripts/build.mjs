import { build, createServer } from 'vite';
import react from '@vitejs/plugin-react';
import ts from 'typescript';
import { readdir, readFile, writeFile, mkdir, cp } from 'node:fs/promises';
import path from 'node:path';
import React from 'react';
import { renderToString } from 'react-dom/server';
const base = '/hidden-marker-lab/';
// Adapt links and requests at compile time; keep evidence files unchanged.
const pathsPlugin = {
 name: 'github-pages-paths', enforce: 'pre',
 transform(source, id) {
  if (!id.includes('/app/') || !id.endsWith('.tsx')) return;
  const file = ts.createSourceFile(id, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const result = ts.transform(file, [context => root => {
   const wrap = value => ts.factory.createCallExpression(ts.factory.createIdentifier('__sitePath'),undefined,[value]);
   const visit = node => {
    if (ts.isJsxAttribute(node) && ['href','src'].includes(node.name.text) && node.initializer) {
     const value = ts.isStringLiteral(node.initializer) ? node.initializer : node.initializer.expression;
     if (value) return ts.factory.updateJsxAttribute(node,node.name,ts.factory.createJsxExpression(undefined,wrap(value)));
    }
    if (ts.isCallExpression(node) && ts.isIdentifier(node.expression) && node.expression.text === 'fetch') return ts.factory.updateCallExpression(node,node.expression,node.typeArguments,[wrap(node.arguments[0]),...node.arguments.slice(1)]);
    return ts.visitEachChild(node,visit,context);
   };
   return ts.visitNode(root,visit);
  }]);
  const output = ts.createPrinter().printFile(result.transformed[0]); result.dispose();
  return `const __sitePath = (url: string) => url.startsWith('/') && !url.startsWith('//') && !url.startsWith('${base}') ? '${base}' + url.slice(1) : url;\n` + output;
 },
};
const routes = [''];
for (const item of await readdir('app',{withFileTypes:true})) {
 if (item.isDirectory()) { try { await readFile(`app/${item.name}/page.tsx`); routes.push(item.name); } catch {} }
}
const imports = routes.map((route,i) => `import Page${i} from './app/${route ? route+'/' : ''}page';`).join('\n');
await writeFile('entry.tsx', `import React from 'react';\nimport {hydrateRoot} from 'react-dom/client';\nimport './app/globals.css';\n${imports}\nconst pages = {${routes.map((route,i)=>`${JSON.stringify(route)}: Page${i}`).join(',')}};\nconst Page = pages[document.documentElement.dataset.route as keyof typeof pages];\nhydrateRoot(document.getElementById('root')!, <Page />);\n`);
const plugins = () => [pathsPlugin,react()];
const server = await createServer({configFile:false,plugins:plugins(),server:{middlewareMode:true},appType:'custom'});
const pages = [];
try {
 for (const route of routes) {
  const mod = await server.ssrLoadModule(`/app/${route ? route+'/' : ''}page.tsx`);
  const title = mod.metadata?.title ?? 'Hidden Marker Lab · Study Overview';
  const description = mod.metadata?.description ?? 'A guided report on whether Qwen3-14B can learn to identify and verbalize arbitrary hidden-state interventions.';
  const escape = s => s.replaceAll('&','&amp;').replaceAll('"','&quot;').replaceAll('<','&lt;');
  const html = `<!DOCTYPE html><html lang="en" data-route="${route}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>${escape(title)}</title><meta name="description" content="${escape(description)}"></head><body><div id="root">${renderToString(React.createElement(mod.default))}</div><script type="module" src="/entry.tsx"></script></body></html>`;
  const filename = route ? `${route}/index.html` : 'index.html';
  if(route) await mkdir(route,{recursive:true});
  await writeFile(filename, html); pages.push(path.resolve(filename));
 }
} finally { await server.close(); }
await build({configFile:false,base,plugins:plugins(),build:{outDir:'dist',copyPublicDir:false,rollupOptions:{input:pages}}});
await cp('public','dist',{recursive:true,mode:2});
await writeFile('dist/.nojekyll','');
console.log(`Exported ${routes.length} pages under ${base}`);
