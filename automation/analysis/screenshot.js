#!/usr/bin/env node
/**
 * Screenshot utility for documentation page validation
 * Usage: node screenshot.js <url> <output_path>
 */

const puppeteer = require('puppeteer');

async function takeScreenshot(url, outputPath) {
    const browser = await puppeteer.launch({
        headless: 'new',
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    try {
        const page = await browser.newPage();
        await page.setViewport({ width: 1280, height: 1024 });

        await page.goto(url, {
            waitUntil: 'networkidle2',
            timeout: 30000
        });

        // Wait a bit for any animations/MathJax to render
        await new Promise(resolve => setTimeout(resolve, 2000));

        await page.screenshot({
            path: outputPath,
            fullPage: false
        });

        console.log(`Screenshot saved to: ${outputPath}`);
        return true;
    } catch (error) {
        console.error(`Error taking screenshot: ${error.message}`);
        return false;
    } finally {
        await browser.close();
    }
}

// CLI handling
const args = process.argv.slice(2);
if (args.length < 2) {
    console.log('Usage: node screenshot.js <url> <output_path>');
    console.log('Example: node screenshot.js https://andrewaltimit.github.io/Documentation/ /tmp/home.png');
    process.exit(1);
}

takeScreenshot(args[0], args[1]).then(success => {
    process.exit(success ? 0 : 1);
});
