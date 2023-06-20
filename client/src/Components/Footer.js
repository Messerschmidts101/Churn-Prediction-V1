import React from 'react'
import Header2 from './Header2'

function Footer() {
    return (
        <footer className='text-center text-lg-start bg-light text-muted'>
            <section className='d-flex justify-content-center justify-content-lg-between p-4 border-bottom'>
                <div className='me-5 d-none d-lg-block'>
                    <p>Get the Repo: </p>
                </div>
                <div>
                    <a href="" class="me-4 text-reset">
                        <i class="fab fa-github"></i>
                    </a>
                </div>
            </section>
            <section>
                <div>
                    <Header2>
                        Group 2
                    </Header2>
                    <p>Alarcon, Chastin</p>
                    <p>Ferrer, Bryan Kristoffer</p>
                </div>
            </section>
        </footer>
    )
}

export default Footer